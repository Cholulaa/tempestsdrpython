"""The Tempest reconstruction pipeline.

Port of the orchestration in ``TempestSDR/src/TSDRLibrary.c`` (Martin Marinov,
GPLv3).  The C library ran four threads connected by ring buffers
(device -> decimation -> post-process -> video); here the same stages run
sequentially inside :meth:`TempestProcessor.process`, which is simpler to
reason about and perfectly adequate for offline files and moderate live rates.

Pixel geometry (``set_internal_samplerate`` in the C source)::

    width      = floor(2 * samplerate / (refresh_rate * height))
    pixel_rate = width * height * refresh_rate

The factor of two horizontally over-samples each line so the sync detector and
resampler have sub-pixel information to work with.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import dsp
from .framerate import FrameRateDetector
from .sync import SyncDetector


@dataclass
class ProcessorConfig:
    samplerate: float
    height: int                      # total lines per frame (incl. blanking)
    refresh_rate: float
    autoshift: bool = True
    nearest: bool = False
    motion_blur: float = 0.0         # frame-averaging coefficient in [0, 1)
    lowpass_before_sync: bool = False
    autogain_after: bool = False
    norm_coeff: float = 0.1          # NORMALISATION_LOWPASS_COEFF in the C code
    detect_framerate: bool = False


@dataclass
class TempestProcessor:
    config: ProcessorConfig
    width: int = field(init=False)
    pixel_rate: float = field(init=False)
    pixels_per_sample: float = field(init=False)

    def __post_init__(self) -> None:
        self._resampler = dsp.Resampler(nearest=self.config.nearest)
        self._autogain = dsp.AutoGain()
        self._sync = SyncDetector()
        self._screen: np.ndarray | None = None
        self._pixbuf = np.zeros(0, dtype=np.float32)
        self.frame_detector = (
            FrameRateDetector(self.config.samplerate)
            if self.config.detect_framerate else None
        )
        self._recompute_geometry()

    # -- geometry ----------------------------------------------------------
    def _recompute_geometry(self) -> None:
        cfg = self.config
        real_width = cfg.samplerate / (cfg.refresh_rate * cfg.height)
        self.width = int(2 * real_width)
        if self.width <= 0:
            raise ValueError(
                "Computed frame width is zero; sample rate is too low for the "
                "requested height/refresh rate."
            )
        self.pixel_rate = self.width * cfg.height * cfg.refresh_rate
        self.pixels_per_sample = self.pixel_rate / cfg.samplerate

    @property
    def frame_shape(self) -> tuple[int, int]:
        return (self.config.height, self.width)

    @property
    def pixels_per_frame(self) -> int:
        return self.width * self.config.height

    # -- core --------------------------------------------------------------
    def reset(self) -> None:
        self._resampler.reset()
        self._autogain = dsp.AutoGain()
        self._sync = SyncDetector()
        self._screen = None
        self._pixbuf = np.zeros(0, dtype=np.float32)
        if self.frame_detector is not None:
            self.frame_detector.reset()

    def _post_process(self, frame: np.ndarray) -> np.ndarray:
        """One frame through autogain -> sync -> frame-average (dsp.c order)."""
        cfg = self.config
        if self._screen is None:
            self._screen = np.zeros(self.frame_shape, dtype=np.float32)

        data = frame
        if not cfg.autogain_after:
            data = self._autogain.run(data, cfg.norm_coeff)

        if cfg.lowpass_before_sync:
            self._screen = dsp.time_lowpass(data, self._screen, cfg.motion_blur)
            wprof, hprof = dsp.collapse_vertical_horizontal(self._screen)
            synced = self._sync.run(self._screen, wprof, hprof, cfg.autoshift)
            result = synced
        else:
            wprof, hprof = dsp.collapse_vertical_horizontal(data)
            synced = self._sync.run(data, wprof, hprof, cfg.autoshift)
            self._screen = dsp.time_lowpass(synced, self._screen, cfg.motion_blur)
            result = self._screen

        if cfg.autogain_after:
            result = self._autogain.run(result, cfg.norm_coeff)
        return np.clip(result, 0.0, 1.0).astype(np.float32)

    def process(self, iq: np.ndarray) -> list[np.ndarray]:
        """Feed a block of complex samples, return any completed frames."""
        envelope = dsp.am_demodulate(iq)

        if self.frame_detector is not None:
            self.frame_detector.run(envelope)

        pixels = self._resampler.process(envelope, self.pixels_per_sample)
        if pixels.size:
            self._pixbuf = np.concatenate([self._pixbuf, pixels])

        frames: list[np.ndarray] = []
        ppf = self.pixels_per_frame
        while self._pixbuf.size >= ppf:
            raw = self._pixbuf[:ppf].reshape(self.frame_shape)
            self._pixbuf = self._pixbuf[ppf:]
            frames.append(self._post_process(raw))
        return frames

    def process_stream(self, source, block_frames: float = 1.0):
        """Yield frames from an iterable/callback IQ ``source``.

        ``source`` may be any iterable of complex NumPy arrays (e.g. a
        :class:`~tempestsdr.sources.base.IQSource`).  Frames are yielded as they
        complete.
        """
        for block in source:
            for frame in self.process(block):
                yield frame

    def reconstruct(self, iq: np.ndarray, max_frames: int | None = None) -> list[np.ndarray]:
        """Convenience: process one big buffer and return all frames."""
        frames = self.process(iq)
        if max_frames is not None:
            frames = frames[:max_frames]
        return frames

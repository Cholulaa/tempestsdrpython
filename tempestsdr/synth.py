"""Synthetic compromising-emanation generator.

The original TempestSDR needs an SDR receiver pointed at a real monitor.  To let
the reconstruction pipeline be exercised, demonstrated and unit-tested without
any hardware, this module *forward-models* the emanation: it takes an image,
lays it into a raster frame (with blanking), scans it line by line into a
baseband envelope, amplitude-modulates it onto a carrier offset and adds
complex Gaussian noise.  Feeding the result back through
:class:`~tempestsdr.processor.TempestProcessor` should recover the picture.

This is a teaching/verification aid, not a model of any specific display, and it
is deliberately simple.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SyntheticConfig:
    total_width: int          # pixels per line incl. blanking
    total_height: int         # lines per frame incl. blanking
    refresh_rate: float
    samplerate: float
    visible_fraction: float = 0.8   # fraction of each axis that carries picture
    num_frames: int = 4
    snr_db: float = 10.0
    freq_offset: float = 0.0        # baseband carrier offset (Hz)
    start_offset_pixels: int = 0    # raster start offset, to exercise sync
    mode: str = "amplitude"         # "amplitude" or "edge"
    seed: int | None = 0


def _build_frame(image: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    """Lay ``image`` into a full raster frame; blanking regions stay at zero."""
    from numpy import asarray

    img = asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()

    vis_w = max(1, int(cfg.total_width * cfg.visible_fraction))
    vis_h = max(1, int(cfg.total_height * cfg.visible_fraction))

    # Resize the source image to the visible area with simple index sampling.
    ys = (np.linspace(0, img.shape[0] - 1, vis_h)).astype(np.int64)
    xs = (np.linspace(0, img.shape[1] - 1, vis_w)).astype(np.int64)
    resized = img[np.ix_(ys, xs)]

    frame = np.zeros((cfg.total_height, cfg.total_width), dtype=np.float64)
    frame[:vis_h, :vis_w] = resized
    return frame


def generate(image: np.ndarray, cfg: SyntheticConfig) -> np.ndarray:
    """Return synthetic complex baseband IQ for ``image``."""
    rng = np.random.default_rng(cfg.seed)
    frame = _build_frame(image, cfg)

    if cfg.mode == "edge":
        # Emphasise horizontal edges, mimicking dV/dt emanations from the cable.
        grad = np.abs(np.diff(frame, axis=1, prepend=frame[:, :1]))
        scan_frame = grad
    else:
        scan_frame = frame

    pixel_seq = scan_frame.reshape(-1)                      # row-major raster scan
    pixel_seq = np.roll(pixel_seq, cfg.start_offset_pixels)
    tiled = np.tile(pixel_seq, cfg.num_frames)

    pixel_rate = cfg.total_width * cfg.total_height * cfg.refresh_rate
    n_samples = int(round(tiled.size * cfg.samplerate / pixel_rate))
    if n_samples < 2:
        raise ValueError("sample rate too low to represent even one frame")

    # Envelope at the receiver sample rate (linear interpolation of pixels).
    pix_pos = np.arange(n_samples) * (pixel_rate / cfg.samplerate)
    envelope = np.interp(pix_pos, np.arange(tiled.size), tiled)

    # Amplitude-modulate onto the carrier offset.
    t = np.arange(n_samples) / cfg.samplerate
    carrier = np.exp(2j * np.pi * cfg.freq_offset * t)
    signal = envelope * carrier

    # Additive white Gaussian noise for the requested SNR.
    sig_power = float(np.mean(np.abs(signal) ** 2))
    if sig_power > 0 and np.isfinite(cfg.snr_db):
        snr_linear = 10.0 ** (cfg.snr_db / 10.0)
        noise_power = sig_power / snr_linear
        sigma = np.sqrt(noise_power / 2.0)
        noise = sigma * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        signal = signal + noise

    return signal.astype(np.complex64)


def load_image(path: str) -> np.ndarray:
    """Load an image file as a 2-D grayscale float array (needs Pillow)."""
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Pillow is required to load image files (pip install pillow)") from exc
    with Image.open(path) as im:
        return np.asarray(im.convert("L"), dtype=np.float64)

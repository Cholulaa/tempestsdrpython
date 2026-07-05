"""Frame-rate and line-rate estimation by FFT autocorrelation.

Port of ``TempestSDR/src/frameratedetector.c`` and the autocorrelation helper
in ``fft.c`` (Martin Marinov, GPLv3).

The idea, quoting the original source: for the demodulated envelope ``v`` we
want the lag ``j`` with the highest autocorrelation ``R(j)``.  A peak at lag
``j`` means the signal repeats every ``j`` samples; a peak in the "frame" lag
band gives the vertical refresh period and a peak in the "line" band gives the
horizontal line period.

Faithfulness note: the C comment in ``frameratedetector.c`` describes the
textbook Wiener-Khinchin power-spectrum form ``R = IFFT(|FFT(v)|**2)``, but the
code it actually runs (``fft_complex_to_absolute_complex`` in ``fft.c``) takes
the *magnitude* spectrum ``sqrt(I**2 + Q**2)`` before the inverse transform, so
the real computation is ``R = |IFFT(|FFT(v)|)|``.  Using the magnitude rather
than the power whitens the spectrum and changes which lag wins for noisy or
multi-tone signals, so this port reproduces the magnitude form to match the
original tool's behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Search bounds copied verbatim from frameratedetector.c
MIN_FRAMERATE = 55.0
MAX_FRAMERATE = 87.0
MIN_HEIGHT = 590
MAX_HEIGHT = 1500
FRAMES_TO_CAPTURE = 3.1  # analysis window length, in slowest-frame units (C: FRAMES_TO_CAPTURE)


def _largest_power_of_two(size: int) -> int:
    """Match ``fft_getrealsize``: the largest power of two <= ``size``."""
    m = 0
    s = size
    while s // 2 != 0:
        s //= 2
        m += 1
    return 1 << m if size > 0 else 0


def autocorrelation(data: np.ndarray) -> np.ndarray:
    """Return the (real) autocorrelation of ``data`` via FFT.

    The input is truncated to the largest power-of-two length, exactly as the C
    implementation does, then ``R = |IFFT(|FFT(x)|)|`` using the *magnitude*
    spectrum, matching ``fft_autocorrelation``/``fft_complex_to_absolute_complex``
    in the original ``fft.c`` (see the module docstring).  ``R[0]`` is the
    zero-lag term; ``R[j]`` the correlation at lag ``j``.
    """
    data = np.asarray(data, dtype=np.float64)
    n = _largest_power_of_two(data.size)
    if n < 2:
        return np.zeros(0, dtype=np.float64)
    x = data[:n]
    spectrum = np.fft.fft(x)
    magnitude = np.abs(spectrum)         # C uses sqrt(I^2+Q^2), not I^2+Q^2
    corr = np.fft.ifft(magnitude)
    return np.abs(corr)


@dataclass
class FrameRateEstimate:
    frame_period_samples: float
    refresh_rate: float
    correlation: float


@dataclass
class LineEstimate:
    line_period_samples: float
    correlation: float


@dataclass
class FrameRateDetector:
    """Accumulate autocorrelations and read off frame/line rates.

    ``run`` may be called repeatedly with successive blocks of the demodulated
    envelope; the autocorrelations are averaged (as in the C ``accummulate``
    routine) to suppress noise before the peaks are located.

    The search bands default to the values baked into the C library (refresh in
    55-87 Hz, total height 590-1500 lines), which cover ordinary computer
    monitors.  They are exposed as fields so unusual displays — or small
    synthetic test signals — can widen them.
    """

    samplerate: float
    min_framerate: float = MIN_FRAMERATE
    max_framerate: float = MAX_FRAMERATE
    min_height: int = MIN_HEIGHT
    max_height: int = MAX_HEIGHT
    window: int | None = None      # analysis window in samples (auto if None)
    _accum: np.ndarray = field(default=None, repr=False)
    _calls: int = 0
    _buffer: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.window is None:
            desired = FRAMES_TO_CAPTURE * self.samplerate / self.min_framerate
            self.window = _largest_power_of_two(int(desired))
        self._buffer = np.zeros(0, dtype=np.float64)

    def reset(self) -> None:
        self._accum = None
        self._calls = 0
        self._buffer = np.zeros(0, dtype=np.float64)

    def run(self, envelope: np.ndarray) -> None:
        """Feed demodulated envelope samples.

        Samples are buffered and analysed in fixed-size windows so that every
        autocorrelation has an identical length and can be averaged, exactly as
        the C frame-rate thread reads a fixed ``desiredsize`` block each time.
        """
        if self.window is None or self.window < 2:
            return
        self._buffer = np.concatenate([self._buffer, np.asarray(envelope, dtype=np.float64)])
        while self._buffer.size >= self.window:
            self._analyse(self._buffer[: self.window])
            self._buffer = self._buffer[self.window:]

    def _analyse(self, window: np.ndarray) -> None:
        corr = autocorrelation(window)
        if corr.size == 0:
            return
        if self._accum is None or self._accum.size != corr.size:
            self._accum = corr.copy()
            self._calls = 1
        else:
            # Running average, matching accummulate()'s (prev*(k-1)+now)/k.
            self._calls += 1
            self._accum += (corr - self._accum) / self._calls

    def _ensure_analysed(self) -> None:
        """If no full window completed, fall back to the buffered remainder."""
        if self._accum is None and self._buffer.size >= 2:
            self._analyse(self._buffer)

    @property
    def correlation(self) -> np.ndarray:
        return self._accum if self._accum is not None else np.zeros(0)

    def _peak_in_band(self, lo_lag: int, hi_lag: int) -> tuple[int, float]:
        corr = self.correlation
        lo_lag = max(1, int(lo_lag))
        hi_lag = min(int(hi_lag), corr.size - 1)
        if corr.size == 0 or hi_lag <= lo_lag:
            return -1, 0.0
        band = corr[lo_lag:hi_lag]
        rel = int(np.argmax(band))
        return lo_lag + rel, float(band[rel])

    def estimate_framerate(self) -> FrameRateEstimate | None:
        self._ensure_analysed()
        lo = self.samplerate / self.max_framerate
        hi = self.samplerate / self.min_framerate
        lag, corr = self._peak_in_band(int(lo), int(hi))
        if lag < 0:
            return None
        return FrameRateEstimate(
            frame_period_samples=float(lag),
            refresh_rate=self.samplerate / lag,
            correlation=corr,
        )

    def estimate_line(self) -> LineEstimate | None:
        self._ensure_analysed()
        lo = self.samplerate / (self.max_height * self.max_framerate)
        hi = self.samplerate / (self.min_height * self.min_framerate)
        lag, corr = self._peak_in_band(int(lo), int(hi))
        if lag < 0:
            return None
        return LineEstimate(line_period_samples=float(lag), correlation=corr)

    def estimate_resolution(self) -> dict | None:
        """Combine frame and line peaks into (refresh rate, total line count)."""
        frame = self.estimate_framerate()
        line = self.estimate_line()
        if frame is None or line is None:
            return None
        height = frame.frame_period_samples / line.line_period_samples
        return {
            "refresh_rate": frame.refresh_rate,
            "frame_period_samples": frame.frame_period_samples,
            "line_period_samples": line.line_period_samples,
            "height_lines": height,
        }

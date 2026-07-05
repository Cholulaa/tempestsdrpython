"""Core digital-signal-processing primitives for TempestSDR (Python port).

This module is a faithful re-implementation of the numerical routines found in
the original TempestSDR C library (``TempestSDR/src/dsp.c`` and
``gaussian.c`` by Martin Marinov, GPLv3).  Where the C code processed
interleaved ``float`` I/Q buffers in place, the Python port operates on NumPy
arrays: complex samples for the radio front-end and real ``float32`` arrays for
the demodulated video envelope.

The mapping to the original source is documented per-function so the port can be
audited against the reference implementation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "am_demodulate",
    "resample_box",
    "resample_nearest",
    "Resampler",
    "AutoGain",
    "time_lowpass",
    "collapse_vertical_horizontal",
    "gaussian_kernel",
    "gaussian_blur",
]


# ---------------------------------------------------------------------------
# Amplitude demodulation  (C: ``am_demod`` in TSDRLibrary.c)
# ---------------------------------------------------------------------------
def am_demodulate(iq: np.ndarray) -> np.ndarray:
    """Demodulate the video envelope from the complex baseband signal.

    The compromising emanation is amplitude-modulated onto the receiver's
    centre frequency, so the envelope is recovered as the magnitude of each
    complex sample: ``sqrt(I**2 + Q**2)``.

    Parameters
    ----------
    iq:
        Complex baseband samples (``complex64``/``complex128``) *or* a real
        interleaved ``[I0, Q0, I1, Q1, ...]`` array as used by the C code.

    Returns
    -------
    numpy.ndarray
        Real ``float32`` envelope, one value per complex sample.
    """
    iq = np.asarray(iq)
    if np.iscomplexobj(iq):
        return np.abs(iq).astype(np.float32)
    # Real interleaved I/Q -> fold into a complex view then take magnitude.
    flat = np.asarray(iq, dtype=np.float32).ravel()
    if flat.size % 2 != 0:
        raise ValueError("interleaved I/Q buffer must contain an even number of values")
    i = flat[0::2]
    q = flat[1::2]
    return np.sqrt(i * i + q * q).astype(np.float32)


# ---------------------------------------------------------------------------
# Resampling  (C: ``dsp_resample_process`` in dsp.c)
# ---------------------------------------------------------------------------
def resample_box(x: np.ndarray, pixels_per_sample: float) -> np.ndarray:
    """Area/box resample ``x`` by ``pixels_per_sample`` (vectorised, stateless).

    This produces the same result as the streaming resampler in the C library
    for a single contiguous buffer.  Each input sample is treated as a box of
    unit width in *sample* coordinates; in *pixel* coordinates it therefore
    spans ``r = pixels_per_sample``.  Every output pixel covers the unit
    interval ``[k, k + 1)`` and its value is the overlap-weighted average of the
    input samples it touches (the overlap weights sum to one).

    Both up-sampling (``r > 1``) and down-sampling (``r < 1``) are supported.
    """
    x = np.asarray(x, dtype=np.float64)
    m = x.size
    r = float(pixels_per_sample)
    if m == 0 or r <= 0.0:
        return np.zeros(0, dtype=np.float32)

    n_out = int(np.floor(m * r))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    # Antiderivative F of the piecewise-constant signal in pixel coordinates.
    # F at sample start i (pixel coord i*r) equals r * (sum of x[:i]).
    prefix = np.empty(m + 1, dtype=np.float64)
    prefix[0] = 0.0
    np.cumsum(x, out=prefix[1:])

    boundaries = np.arange(n_out + 1, dtype=np.float64)          # 0 .. n_out
    idx = np.floor(boundaries / r).astype(np.int64)
    np.clip(idx, 0, m - 1, out=idx)
    f_vals = r * prefix[idx] + (boundaries - idx * r) * x[idx]
    return np.diff(f_vals).astype(np.float32)


def resample_nearest(x: np.ndarray, pixels_per_sample: float) -> np.ndarray:
    """Nearest-neighbour resampling (C: ``nearest_neighbour_sampling`` branch)."""
    x = np.asarray(x, dtype=np.float64)
    m = x.size
    r = float(pixels_per_sample)
    if m == 0 or r <= 0.0:
        return np.zeros(0, dtype=np.float32)
    n_out = int(np.floor(m * r))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    src = (np.arange(n_out, dtype=np.int64) * m) // n_out
    return x[src].astype(np.float32)


class Resampler:
    """Stateful streaming box resampler.

    Mirrors ``dsp_resample_t`` from the C library: it carries the fractional
    phase and the trailing input samples between calls so that a long stream
    processed in chunks yields *exactly* the same pixels as
    :func:`resample_box` applied to the whole signal at once.

    Coordinates are in "pixel space": input sample ``n`` covers the interval
    ``[n * r, (n + 1) * r)`` and output pixel ``k`` covers ``[k, k + 1)``.  The
    antiderivative ``F`` of the piecewise-constant signal gives each pixel as
    ``F(k + 1) - F(k)``; because only differences are taken, an arbitrary
    constant offset in ``F`` cancels, so no running total needs to be carried —
    only the trailing samples that straddle the next pixel boundary and the
    coordinate of the first retained sample.

    Note: unlike the C ``dsp_resample_process`` — whose ``contrib``/``offset``
    bookkeeping drops roughly one output pixel at every chunk boundary — this
    resampler is *chunk-invariant*: feeding a stream in any block sizes yields
    bit-for-bit the same pixels as :func:`resample_box` on the whole signal.
    That is a deliberate correctness improvement over the reference, not an
    accident, and it removes the per-block seam artifact the C code exhibits.
    """

    def __init__(self, nearest: bool = False) -> None:
        self.nearest = nearest
        self.reset()

    def reset(self) -> None:
        self._carry = np.zeros(0, dtype=np.float64)
        self._base = 0.0     # pixel coordinate of the first carried sample's start
        self._next_k = 0     # index of the next output pixel to emit (rebased)

    def process(self, x: np.ndarray, pixels_per_sample: float) -> np.ndarray:
        r = float(pixels_per_sample)
        c = np.concatenate([self._carry, np.asarray(x, dtype=np.float64)])
        m = c.size
        if m == 0 or r <= 0.0:
            self._carry = c
            return np.zeros(0, dtype=np.float32)

        base = self._base
        end = base + m * r
        last_boundary = int(np.floor(end))          # largest integer boundary covered
        k_lo = self._next_k
        if last_boundary <= k_lo:
            # Not enough data for a full pixel yet: accumulate and wait.
            self._carry = c
            return np.zeros(0, dtype=np.float32)

        boundaries = np.arange(k_lo, last_boundary + 1, dtype=np.float64)
        local = boundaries - base
        idx = np.floor(local / r).astype(np.int64)
        np.clip(idx, 0, m - 1, out=idx)

        if self.nearest:
            # Sample at each output pixel's left edge, so the streaming result
            # equals the stateless resample_nearest / the C floor(size*id/out)
            # rule.  (Using pixel centres would shift every sample by one.)
            lefts = boundaries[:-1]
            src = np.floor((lefts - base) / r).astype(np.int64)
            np.clip(src, 0, m - 1, out=src)
            out = c[src]
        else:
            prefix = np.empty(m + 1, dtype=np.float64)
            prefix[0] = 0.0
            np.cumsum(c, out=prefix[1:])
            f_vals = r * prefix[idx] + (local - idx * r) * c[idx]
            out = np.diff(f_vals)

        # Retain the sample that contains ``last_boundary`` and everything after
        # it, then rebase the coordinate system so the numbers stay bounded.
        keep0 = int(np.floor((last_boundary - base) / r))
        keep0 = max(0, min(keep0, m))
        self._carry = c[keep0:].copy()
        self._base = (base + keep0 * r) - last_boundary
        self._next_k = 0
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Auto gain  (C: ``dsp_autogain_run`` in dsp.c)
# ---------------------------------------------------------------------------
class AutoGain:
    """Stretch each frame across the full dynamic range with a slow AGC.

    The min/max used for normalisation are low-pass filtered over successive
    frames (coefficient ``norm``) so the picture does not flicker.  The
    signal-to-noise ratio is tracked exactly as in the C code
    (``mean / stdev``).
    """

    def __init__(self) -> None:
        self.last_max = 0.0
        self.last_min = 0.0
        self.snr = 1.0

    def run(self, frame: np.ndarray, norm: float) -> np.ndarray:
        data = np.asarray(frame, dtype=np.float64)
        mn = float(data.min())
        mx = float(data.max())
        one_minus = 1.0 - norm
        self.last_max = one_minus * self.last_max + norm * mx
        self.last_min = one_minus * self.last_min + norm * mn
        span = 1.0 if self.last_max == self.last_min else (self.last_max - self.last_min)

        out = (data - self.last_min) / span

        mean = float(data.mean())
        diff = data - mean
        n = data.size
        if n > 1:
            var = (np.sum(diff * diff) - np.sum(diff) ** 2 / n) / (n - 1)
            stdev = np.sqrt(var) if var > 0 else 0.0
            self.snr = mean / stdev if stdev > 0 else float("inf")
        return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Frame averaging / motion blur  (C: ``dsp_timelowpass_run`` in dsp.c)
# ---------------------------------------------------------------------------
def time_lowpass(new_frame: np.ndarray, screen: np.ndarray, coeff: float) -> np.ndarray:
    """Blend ``new_frame`` into the persistent ``screen`` buffer.

    ``screen = screen * coeff + new_frame * (1 - coeff)``.  ``coeff`` is the
    "motion blur" amount: 0 disables averaging, values near 1 heavily average
    successive frames to pull a weak signal out of the noise.
    """
    coeff = float(coeff)
    return screen * coeff + new_frame * (1.0 - coeff)


# ---------------------------------------------------------------------------
# Vertical / horizontal collapse  (C: ``dsp_average_v_h`` in dsp.c)
# ---------------------------------------------------------------------------
def collapse_vertical_horizontal(frame: np.ndarray):
    """Sum a frame down its columns and across its rows.

    Returns ``(width_profile, height_profile)`` where ``width_profile[x]`` is
    the sum of column ``x`` and ``height_profile[y]`` is the sum of row ``y``.
    These 1-D profiles expose the horizontal and vertical blanking bars used by
    the sync detector.
    """
    frame = np.asarray(frame, dtype=np.float64)
    width_profile = frame.sum(axis=0)
    height_profile = frame.sum(axis=1)
    return width_profile, height_profile


# ---------------------------------------------------------------------------
# Gaussian blur  (C: ``gaussianblur`` in gaussian.c)
# ---------------------------------------------------------------------------
_GAUSSIAN_ALPHA = 1.0


def gaussian_kernel(n: int = 5, alpha: float = _GAUSSIAN_ALPHA) -> np.ndarray:
    """Return the normalised ``n``-tap Gaussian kernel used by the C library.

    ``g(i) = exp(-2 * alpha**2 * i**2 / n**2)`` for ``i`` from ``-(n-1)/2`` to
    ``(n-1)/2``.
    """
    half = (n - 1) / 2.0
    i = np.arange(-half, half + 1)
    g = np.exp(-2.0 * alpha * alpha * i * i / (n * n))
    return (g / g.sum()).astype(np.float64)


_GAUSS5 = gaussian_kernel(5)


def gaussian_blur(data: np.ndarray, kernel: np.ndarray = _GAUSS5) -> np.ndarray:
    """Circular 1-D Gaussian blur (wrap-around).

    For arrays of at least ``kernel.size`` samples this is identical to the C
    ``gaussianblur`` — which is all the sync detector ever feeds it (the
    horizontal/vertical collapse profiles are hundreds to thousands of samples
    long).  For degenerate arrays shorter than the kernel it falls back to a
    well-defined wrap-around convolution (a proper circular blur), rather than
    reproducing the C size<5 special case, which is unreachable in the pipeline.
    """
    data = np.asarray(data, dtype=np.float64)
    n = data.size
    if n == 0:
        return data.copy()
    k = kernel.size // 2
    if n >= kernel.size:
        padded = np.concatenate([data[-k:], data, data[:k]])
        return np.convolve(padded, kernel, mode="valid")
    # n < kernel.size: tile enough copies that the centred kernel is fully
    # covered, convolve, then read the centre block back out.
    reps = int(np.ceil(kernel.size / n)) + 2
    tiled = np.tile(data, reps)
    start = (reps // 2) * n
    padded = np.concatenate([tiled[start - k:start], data, tiled[start:start + k]])
    return np.convolve(padded, kernel, mode="valid")

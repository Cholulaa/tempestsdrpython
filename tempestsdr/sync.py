"""Image synchronisation: locate blanking regions and align the frame.

Port of ``TempestSDR/src/syncdetector.c`` (Martin Marinov, GPLv3).

A reconstructed frame arrives with an arbitrary offset because the receiver has
no idea where the target's raster scan starts.  The blanking intervals (the dark
bars between visible lines/frames) show up as a contiguous low-energy strip in
the horizontal (``width``) and vertical (``height``) collapse profiles.  The
"sweet spot" search finds, for each profile, the circular strip whose mean
differs most from the rest of the signal; its centre is the detected sync
offset.  The offset is low-pass filtered over frames for stability, and its
velocity drives an optional frame-rate PLL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .dsp import gaussian_blur


def _c_round(x: float) -> int:
    """Round half away from zero, matching C's ``round()`` (not banker's)."""
    return int(math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5))

# Coefficients copied from syncdetector.c
FRAMERATE_DX_LOWPASS_COEFF_HEIGHT = 0.1
FRAMERATE_DX_LOWPASS_COEFF_WIDTH = 0.9
FRAMERATE_PLL_SPEED_HI = 0.00001
FRAMERATE_PLL_SPEED_LO = 0.000001
FRAMERATE_PLL_LOCKED_VALUE = 0.5


@dataclass
class SweetSpot:
    """State for one axis of the sync detector (C: ``sweetspot_data_t``)."""

    curr_stripsize: int = 0
    dx: int = 0       # low-pass filtered centre of the blanking strip
    vx: int = 0       # per-frame velocity of ``dx``
    absvx: int = 0


def _find_best_fit(data: np.ndarray, total_sum: float, stripsize: int):
    """Return ``(best_fit, best_start)`` for a circular strip of ``stripsize``.

    For every start index the fitness is the squared difference between the mean
    inside the strip and the mean outside it; the maximum marks the blanking
    bar.  Vectorised equivalent of ``findbestfit``.

    Faithfulness note: the C ``findbestfit`` records ``bestfitid = i`` at the
    point where its incrementally-updated sum already corresponds to the window
    starting at ``i + 1``, so a strip that truly starts at ``s >= 1`` is
    reported at ``s - 1``.  We reproduce that off-by-one exactly so the detected
    sync offset (and the autoshift roll) matches the original tool to the pixel.
    """
    size = data.size
    if stripsize < 1 or stripsize >= size:
        return -np.inf, 0
    big = size - stripsize
    # Circular window sums via a doubled prefix-sum.
    doubled = np.concatenate([data, data[: stripsize]])
    csum = np.concatenate([[0.0], np.cumsum(doubled)])
    window = csum[stripsize : stripsize + size] - csum[0:size]  # sum over [i, i+stripsize)
    fit = (total_sum - window) / big - window / stripsize
    fit = fit * fit
    true_start = int(np.argmax(fit))        # first index achieving the maximum
    c_start = true_start - 1 if true_start >= 1 else 0
    return float(fit[true_start]), c_start


def find_the_sweet_spot(db: SweetSpot, data: np.ndarray, minsize: int, lowpass: float) -> np.ndarray:
    """Locate the blanking strip in a 1-D collapse profile.

    Mutates ``db`` (updates ``curr_stripsize``, ``dx``, ``vx``, ``absvx``) and
    returns the Gaussian-blurred profile.  Follows the C ``findthesweetspot``
    routine, including the four candidate strip sizes it probes each frame.
    """
    data = np.array(data, dtype=np.float64)
    size = data.size
    if minsize < 1:
        minsize = 1
    size2 = size >> 1

    if db.curr_stripsize < minsize:
        db.curr_stripsize = minsize
    elif db.curr_stripsize > size2:
        db.curr_stripsize = size2

    data = gaussian_blur(data)
    total_sum = float(data.sum())

    best_fit, best_start = _find_best_fit(data, total_sum, db.curr_stripsize)
    best_size = db.curr_stripsize

    for candidate in (
        db.curr_stripsize - 4,
        db.curr_stripsize + 4,
        db.curr_stripsize >> 1,
        db.curr_stripsize << 1,
    ):
        if minsize <= candidate < size2 and candidate != db.curr_stripsize:
            fit, start = _find_best_fit(data, total_sum, candidate)
            if fit > best_fit:
                best_fit, best_start, best_size = fit, start, candidate

    db.curr_stripsize = best_size

    # Unwrap so the low-pass filter takes the shortest path around the circle.
    # This mirrors the C ``findthesweetspot`` exactly, including that ``last_x``
    # captures the (possibly ``+size``) adjusted ``dx`` so the velocity ``vx``
    # keeps the right sign across a wrap-around.
    h2 = size // 2
    dx_no_lowpass = (best_start + best_size // 2) % size
    raw_diff = dx_no_lowpass - db.dx
    if raw_diff > h2:
        db.dx += size
    elif raw_diff < -h2:
        dx_no_lowpass += size

    last_x = db.dx
    db.dx = _c_round(dx_no_lowpass * lowpass + (1.0 - lowpass) * db.dx) % size

    raw_vx = db.dx - last_x
    if raw_vx > h2:
        db.vx = int(size - raw_vx)
    elif raw_vx < -h2:
        db.vx = int(-size - raw_vx)
    else:
        db.vx = int(raw_vx)
    db.absvx = abs(db.vx)
    return data


@dataclass
class SyncDetector:
    """Two-axis sync detector plus an optional frame-rate PLL."""

    db_x: SweetSpot = None
    db_y: SweetSpot = None
    avg_speed: float = 0.0
    locked: bool = False

    def __post_init__(self) -> None:
        if self.db_x is None:
            self.db_x = SweetSpot()
        if self.db_y is None:
            self.db_y = SweetSpot()

    def run(
        self,
        frame: np.ndarray,
        width_profile: np.ndarray,
        height_profile: np.ndarray,
        autoshift: bool = True,
    ) -> np.ndarray:
        """Return the sync-corrected frame.

        With ``autoshift`` the frame is circularly rolled so the detected
        blanking moves to the top-left origin (C: ``PARAM_INT_AUTOSHIFT``).
        Without it the input frame is returned unchanged (the caller may still
        read ``db_x.dx`` / ``db_y.dx`` to draw sync guides).
        """
        find_the_sweet_spot(
            self.db_x, width_profile, int(width_profile.size * 0.05),
            FRAMERATE_DX_LOWPASS_COEFF_WIDTH,
        )
        find_the_sweet_spot(
            self.db_y, height_profile, int(height_profile.size * 0.01),
            FRAMERATE_DX_LOWPASS_COEFF_HEIGHT,
        )

        # Track lock state exactly as frameratepll() does.
        self.avg_speed = self.avg_speed * 0.99 + 0.01 * self.db_x.vx
        self.locked = -FRAMERATE_PLL_LOCKED_VALUE < self.avg_speed < FRAMERATE_PLL_LOCKED_VALUE

        if autoshift:
            return np.roll(frame, shift=(-self.db_y.dx, -self.db_x.dx), axis=(0, 1))
        return frame

    def pll_frequency_correction(self) -> float:
        """Return the refresh-rate delta the PLL would apply this frame.

        The caller adjusts ``refreshrate -= delta`` and recomputes the sampling
        geometry (the C code does this inside ``frameratepll``).  Returns 0 when
        the loop is disabled or the offset is stationary.
        """
        if self.db_x.vx == 0:
            return 0.0
        if not self.locked:
            return self.db_x.vx * FRAMERATE_PLL_SPEED_HI
        return self.avg_speed * FRAMERATE_PLL_SPEED_LO

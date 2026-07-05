"""Tests for the blanking/sweet-spot sync detector."""

import numpy as np

from tempestsdr.sync import SweetSpot, SyncDetector, find_the_sweet_spot, _find_best_fit


def test_find_best_fit_locates_dark_strip():
    # A profile that is high everywhere except a contiguous low "blanking" strip.
    size = 100
    data = np.ones(size) * 10.0
    data[60:75] = 0.0                 # blanking strip of width 15 starting at 60
    total = float(data.sum())
    fit, start = _find_best_fit(data, total, 15)
    # The port reproduces the C findbestfit off-by-one: a strip truly starting at
    # 60 is reported at 59 (see _find_best_fit docstring).
    assert start == 59


def test_sweet_spot_center_tracks_blanking():
    size = 200
    data = np.ones(size)
    data[30:50] = 0.0                 # strip centred near 40
    db = SweetSpot()
    # Run several times so the low-pass filter settles.
    for _ in range(30):
        find_the_sweet_spot(db, data, minsize=int(size * 0.05), lowpass=0.9)
    assert 30 <= db.dx <= 50


def test_autoshift_moves_blanking_to_origin():
    # Build a frame whose dark blanking bars are offset into the middle;
    # after autoshift the blanking should sit at the top-left edges.
    h, w = 60, 80
    frame = np.ones((h, w), dtype=np.float32)
    frame[25:30, :] = 0.0             # horizontal blanking band at rows 25-29
    frame[:, 40:46] = 0.0             # vertical blanking band at cols 40-45

    sync = SyncDetector()
    wprof = frame.sum(axis=0)
    hprof = frame.sum(axis=1)
    out = None
    for _ in range(40):               # let the low-pass converge
        wprof = frame.sum(axis=0)
        hprof = frame.sum(axis=1)
        out = sync.run(frame, wprof, hprof, autoshift=True)

    # The darkest row/column of the shifted frame should be near the top/left.
    row_energy = out.sum(axis=1)
    col_energy = out.sum(axis=0)
    assert row_energy.argmin() < 12 or row_energy.argmin() > h - 12
    assert col_energy.argmin() < 12 or col_energy.argmin() > w - 12


def test_sweet_spot_velocity_sign_on_wraparound():
    # Regression: when the blanking strip moves backward across the 0/size seam,
    # the reported velocity must be negative (it took the short path), matching
    # the C findthesweetspot.  A naive port flips the sign here.
    size = 100

    def profile(center, width=8):
        d = np.ones(size)
        for i in range(width):
            d[(center - width // 2 + i) % size] = 0.0
        return d

    db = SweetSpot()
    for _ in range(60):
        find_the_sweet_spot(db, profile(5), minsize=int(size * 0.05), lowpass=0.9)
    find_the_sweet_spot(db, profile(95), minsize=int(size * 0.05), lowpass=0.9)
    assert db.vx < 0


def test_no_autoshift_returns_input_unchanged():
    frame = np.random.default_rng(0).random((20, 30)).astype(np.float32)
    sync = SyncDetector()
    out = sync.run(frame, frame.sum(axis=0), frame.sum(axis=1), autoshift=False)
    np.testing.assert_array_equal(out, frame)


def test_pll_correction_zero_when_stationary():
    sync = SyncDetector()
    sync.db_x.vx = 0
    assert sync.pll_frequency_correction() == 0.0

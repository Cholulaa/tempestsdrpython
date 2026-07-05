"""Tests for FFT-autocorrelation frame/line-rate detection."""

import numpy as np
import pytest

from tempestsdr import dsp
from tempestsdr.framerate import FrameRateDetector, autocorrelation, _largest_power_of_two
from tempestsdr.synth import SyntheticConfig, generate


def test_largest_power_of_two():
    assert _largest_power_of_two(1024) == 1024
    assert _largest_power_of_two(1000) == 512
    assert _largest_power_of_two(1025) == 1024
    assert _largest_power_of_two(1) == 1


def test_autocorrelation_peaks_at_period():
    # A repeating non-negative pattern (like a scanned video line) autocorrelates
    # strongly at multiples of its period.  A pure sine is deliberately avoided:
    # the algorithm takes the magnitude of the correlation, so a sine's
    # half-period anti-correlation would be indistinguishable from its peak.
    period = 64
    rng = np.random.default_rng(0)
    pattern = rng.random(period)          # one "line" of non-negative pixels
    x = np.tile(pattern, 64)              # 64 repetitions
    corr = autocorrelation(x)
    # The period lag must dominate non-multiple lags.
    assert corr[period] > corr[period // 2]
    assert corr[period] > corr[period + 7]
    assert corr[period] > corr[period - 7]


def test_detector_recovers_refresh_and_height():
    rng = np.random.default_rng(3)
    img = rng.random((120, 160))
    total_w, total_h, refresh = 160, 130, 60.0
    samplerate = total_w * total_h * refresh / 2
    iq = generate(img, SyntheticConfig(
        total_width=total_w, total_height=total_h, refresh_rate=refresh,
        samplerate=samplerate, num_frames=16, snr_db=20, mode="edge", seed=3))

    det = FrameRateDetector(samplerate, min_framerate=50, max_framerate=70,
                            min_height=100, max_height=160)
    env = dsp.am_demodulate(iq)
    block = 20000
    for i in range(0, env.size, block):
        det.run(env[i:i + block])

    est = det.estimate_resolution()
    assert est is not None
    assert est["refresh_rate"] == pytest.approx(refresh, abs=1.0)
    assert est["height_lines"] == pytest.approx(total_h, abs=15)


def test_detector_is_block_size_invariant():
    # Detection must not depend on how the stream is chopped into run() calls.
    rng = np.random.default_rng(4)
    img = rng.random((100, 120))
    total_w, total_h, refresh = 140, 120, 60.0
    sr = total_w * total_h * refresh / 2
    iq = generate(img, SyntheticConfig(
        total_width=total_w, total_height=total_h, refresh_rate=refresh,
        samplerate=sr, num_frames=16, snr_db=25, mode="edge", seed=5))
    env = dsp.am_demodulate(iq)

    results = []
    for block in (9999, 33333, env.size):
        det = FrameRateDetector(sr, min_framerate=50, max_framerate=70,
                                min_height=90, max_height=150)
        for i in range(0, env.size, block):
            det.run(env[i:i + block])
        est = det.estimate_framerate()
        results.append(est.refresh_rate)
    assert max(results) - min(results) < 0.5

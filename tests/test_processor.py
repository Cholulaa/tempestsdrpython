"""Tests for the processor's live-reconfiguration logic."""

import numpy as np
import pytest

from tempestsdr.processor import ProcessorConfig, TempestProcessor


def _proc(**kw):
    cfg = ProcessorConfig(samplerate=8_000_000, height=628, refresh_rate=60.0, **kw)
    return TempestProcessor(cfg)


def test_reconfigure_param_only_no_geometry_change():
    p = _proc()
    w0, pps0 = p.width, p.pixels_per_sample
    p.reconfigure(motion_blur=0.7, autoshift=False)
    assert p.config.motion_blur == 0.7
    assert p.config.autoshift is False
    assert p.width == w0 and p.pixels_per_sample == pps0


def test_reconfigure_geometry_recomputes_and_resets():
    p = _proc()
    # push some pixels into the buffer, then change geometry
    p.process((np.ones(200_000) + 0j).astype(np.complex64))
    p.reconfigure(height=806, refresh_rate=70.0)
    assert p.config.height == 806 and p.config.refresh_rate == 70.0
    expected_w = int(2 * 8_000_000 / (70.0 * 806))
    assert p.width == expected_w
    assert p._pixbuf.size == 0  # reset cleared the buffer


def test_reconfigure_nearest_rebuilds_resampler():
    p = _proc()
    assert p._resampler.nearest is False
    p.reconfigure(nearest=True)
    assert p._resampler.nearest is True


def test_reconfigure_unknown_field_raises():
    p = _proc()
    with pytest.raises(AttributeError):
        p.reconfigure(does_not_exist=1)


def test_framerate_pll_nudges_pixel_rate_without_changing_width():
    p = _proc(framerate_pll=True)
    p._sync.db_x.vx = 5          # pretend the blanking is drifting
    p._sync.avg_speed = 5.0
    w0 = p.width
    r0 = p.config.refresh_rate
    p._apply_pll()
    assert p.width == w0                 # width is held constant
    assert p.config.refresh_rate != r0   # refresh nudged
    assert p.pixels_per_sample == p.pixel_rate / p.config.samplerate

"""End-to-end reconstruction tests: image -> synthetic capture -> image.

These are the acceptance tests for the whole toolkit.  A known picture is
forward-modelled into a synthetic emanation, run through the reconstruction
pipeline, and the recovered frame is compared to the ground truth with an
alignment-invariant (circular-shift) normalised cross-correlation, since the
sync offset and start phase are arbitrary.
"""

import numpy as np
import pytest

from tempestsdr.processor import ProcessorConfig, TempestProcessor
from tempestsdr.sources.file_source import FileSource, save_iq
from tempestsdr.synth import SyntheticConfig, generate, _build_frame


def _max_xcorr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    xc = np.fft.ifft2(np.fft.fft2(a) * np.conj(np.fft.fft2(b))).real
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float(xc.max() / denom) if denom > 0 else 0.0


def _test_pattern(h=120, w=160):
    img = np.zeros((h, w))
    img[10:40, 10:w - 10] = 1.0
    img[10:h - 10, 10:30] = 1.0
    img[h // 2:h // 2 + 20, 10:w - 40] = 0.6
    img[h - 40:h - 10, 40:w - 10] = 0.9
    img[h - 20:h, :] = np.linspace(0, 1, w)
    return img


@pytest.fixture
def geometry():
    total_w, total_h, refresh = 200, 150, 60.0
    samplerate = total_w * total_h * refresh / 2  # so processor width == total_w
    return total_w, total_h, refresh, samplerate


@pytest.mark.parametrize("snr_db,threshold", [(20, 0.85), (10, 0.85), (0, 0.7)])
def test_reconstruction_quality(geometry, snr_db, threshold):
    total_w, total_h, refresh, sr = geometry
    img = _test_pattern()
    cfg = SyntheticConfig(total_width=total_w, total_height=total_h, refresh_rate=refresh,
                          samplerate=sr, num_frames=12, snr_db=snr_db,
                          start_offset_pixels=3777, seed=1)
    iq = generate(img, cfg)
    proc = TempestProcessor(ProcessorConfig(
        samplerate=sr, height=total_h, refresh_rate=refresh, motion_blur=0.6))
    frames = proc.process(iq)
    assert frames, "no frames reconstructed"
    ref = _build_frame(img, cfg)
    assert _max_xcorr(frames[-1], ref) >= threshold


def test_processor_geometry(geometry):
    total_w, total_h, refresh, sr = geometry
    proc = TempestProcessor(ProcessorConfig(samplerate=sr, height=total_h, refresh_rate=refresh))
    assert proc.width == total_w
    assert proc.pixels_per_sample == pytest.approx(2.0)
    assert proc.pixels_per_frame == total_w * total_h


def test_chunked_equals_whole_buffer(geometry):
    total_w, total_h, refresh, sr = geometry
    img = _test_pattern()
    cfg = SyntheticConfig(total_width=total_w, total_height=total_h, refresh_rate=refresh,
                          samplerate=sr, num_frames=8, snr_db=15, start_offset_pixels=1000, seed=2)
    iq = generate(img, cfg)

    p_whole = TempestProcessor(ProcessorConfig(samplerate=sr, height=total_h,
                                               refresh_rate=refresh, motion_blur=0.5))
    whole = p_whole.process(iq)

    p_chunk = TempestProcessor(ProcessorConfig(samplerate=sr, height=total_h,
                                               refresh_rate=refresh, motion_blur=0.5))
    chunked = []
    for i in range(0, iq.size, 7777):
        chunked.extend(p_chunk.process(iq[i:i + 7777]))

    assert len(whole) == len(chunked)
    # Identical inputs, identical (deterministic) pipeline -> identical output.
    np.testing.assert_allclose(whole[-1], chunked[-1], atol=1e-5)


def test_full_pipeline_via_file(geometry, tmp_path):
    total_w, total_h, refresh, sr = geometry
    img = _test_pattern()
    cfg = SyntheticConfig(total_width=total_w, total_height=total_h, refresh_rate=refresh,
                          samplerate=sr, num_frames=10, snr_db=15, start_offset_pixels=500, seed=7)
    iq = generate(img, cfg)

    path = tmp_path / "capture.iq"
    save_iq(str(path), iq, sample_format="uint8")

    src = FileSource(str(path), samplerate=sr, sample_format="uint8")
    proc = TempestProcessor(ProcessorConfig(samplerate=sr, height=total_h,
                                            refresh_rate=refresh, motion_blur=0.6))
    frames = []
    for block in src:
        frames.extend(proc.process(block))
    assert frames
    ref = _build_frame(img, cfg)
    # uint8 quantisation costs a little quality but the picture must survive.
    assert _max_xcorr(frames[-1], ref) >= 0.7

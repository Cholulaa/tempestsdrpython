"""Tests for the core DSP primitives, including faithfulness to the C code."""

import numpy as np
import pytest

from tempestsdr import dsp


def _c_resample_reference(x, r):
    """Literal scalar port of ``dsp_resample_process`` (non-NN branch).

    Used as an oracle for the vectorised :func:`dsp.resample_box`.
    """
    out = []
    contrib = 0.0
    pid = 0
    stp = r  # sampletimeoverpixel
    for i, val in enumerate(x):
        idcheck = i * stp
        idcheck3 = idcheck + stp
        idcheck2 = idcheck + stp - 1.0
        if pid < idcheck and pid < idcheck2:
            out.append(contrib + val * (1.0 - idcheck + pid))
            contrib = 0.0
            pid += 1
        while pid < idcheck2:
            out.append(val)
            pid += 1
        if idcheck < pid < idcheck3:
            contrib += (idcheck3 - pid) * val
        else:
            contrib += stp * val
    return np.array(out)


def test_am_demodulate_complex():
    iq = np.array([3 + 4j, 0 + 0j, 1 + 0j, 0 + 1j], dtype=np.complex64)
    np.testing.assert_allclose(dsp.am_demodulate(iq), [5.0, 0.0, 1.0, 1.0], atol=1e-6)


def test_am_demodulate_interleaved_matches_complex():
    rng = np.random.default_rng(0)
    iq = (rng.standard_normal(1000) + 1j * rng.standard_normal(1000)).astype(np.complex64)
    inter = np.empty(2000, dtype=np.float32)
    inter[0::2] = iq.real
    inter[1::2] = iq.imag
    np.testing.assert_allclose(dsp.am_demodulate(inter), dsp.am_demodulate(iq), atol=1e-5)


def test_am_demodulate_rejects_odd_interleaved():
    with pytest.raises(ValueError):
        dsp.am_demodulate(np.zeros(5, dtype=np.float32))


@pytest.mark.parametrize("r", [0.5, 1.0, 1.3, 2.0, 3.7, 1.618])
def test_resample_box_matches_c_reference(r):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    box = dsp.resample_box(x, r)
    ref = _c_resample_reference(x, r)
    n = min(len(box), len(ref))
    assert abs(len(box) - len(ref)) <= 1  # off-by-one at the very tail only
    np.testing.assert_allclose(box[:n], ref[:n], atol=1e-6)


def test_resample_box_length():
    x = np.ones(1000)
    assert dsp.resample_box(x, 2.0).size == 2000
    assert dsp.resample_box(x, 0.5).size == 500


def test_resample_box_preserves_dc():
    # A constant signal must remain constant through box resampling.
    x = np.full(500, 0.7)
    out = dsp.resample_box(x, 1.3)
    np.testing.assert_allclose(out, 0.7, atol=1e-6)


@pytest.mark.parametrize("r", [0.5, 1.0, 1.3, 2.0, 3.7])
@pytest.mark.parametrize("chunks", [[2000], [123, 4567, 7, 999], [1] * 200 + [1800]])
def test_streaming_resampler_matches_batch(r, chunks):
    rng = np.random.default_rng(2)
    total = sum(chunks)
    x = rng.standard_normal(total)
    batch = dsp.resample_box(x, r)
    rs = dsp.Resampler()
    out, i = [], 0
    for sz in chunks:
        out.append(rs.process(x[i:i + sz], r))
        i += sz
    stream = np.concatenate(out)
    assert stream.size == batch.size
    np.testing.assert_allclose(stream, batch, atol=1e-6)


def test_resample_nearest_length_and_values():
    x = np.arange(10.0)
    out = dsp.resample_nearest(x, 2.0)
    assert out.size == 20
    # nearest neighbour keeps original sample values
    assert set(np.unique(out)).issubset(set(x))


@pytest.mark.parametrize("r", [0.5, 1.0, 2.0, 1.3])
def test_streaming_nearest_whole_buffer_matches_stateless(r):
    # Processed in one call, the streaming nearest path must agree exactly with
    # the stateless resample_nearest (the C floor(size*id/out) rule).  This is
    # the fix for the original center-of-pixel off-by-one.
    x = np.arange(1.0, 61.0)
    batch = dsp.resample_nearest(x, r)
    rs = dsp.Resampler(nearest=True)
    stream = rs.process(x, r)
    np.testing.assert_array_equal(stream, batch)


@pytest.mark.parametrize("chunks", [[7, 13, 40], [1] * 60])
def test_streaming_nearest_chunked_close_to_stateless(chunks):
    # Nearest-neighbour resampling across chunk boundaries may pick an adjacent
    # input sample (a <=1-sample choice), so chunked NN is allowed to differ by
    # one sample index from the whole-buffer result at boundaries.  (The default
    # box resampler, by contrast, is bit-exact across chunk boundaries.)
    r = 1.3
    total = sum(chunks)
    x = np.arange(1.0, total + 1.0)
    batch = dsp.resample_nearest(x, r)
    rs = dsp.Resampler(nearest=True)
    out, i = [], 0
    for sz in chunks:
        out.append(rs.process(x[i:i + sz], r))
        i += sz
    stream = np.concatenate(out)
    n = min(stream.size, batch.size)
    assert abs(stream.size - batch.size) <= 1
    # x is arange, so a one-sample index difference shows up as a value diff of 1.
    assert np.max(np.abs(stream[:n] - batch[:n])) <= 1.0


def test_autogain_spreads_to_unit_range():
    ag = dsp.AutoGain()
    frame = np.linspace(2.0, 5.0, 100).reshape(10, 10).astype(np.float32)
    out = ag.run(frame, norm=1.0)  # norm=1 -> use this frame's min/max directly
    assert out.min() == pytest.approx(0.0, abs=1e-6)
    assert out.max() == pytest.approx(1.0, abs=1e-6)


def test_time_lowpass_blends():
    screen = np.zeros((4, 4))
    new = np.ones((4, 4))
    out = dsp.time_lowpass(new, screen, coeff=0.5)
    np.testing.assert_allclose(out, 0.5)


def test_collapse_vertical_horizontal():
    frame = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    wprof, hprof = dsp.collapse_vertical_horizontal(frame)
    np.testing.assert_allclose(wprof, [5.0, 7.0, 9.0])   # column sums
    np.testing.assert_allclose(hprof, [6.0, 15.0])       # row sums


def test_gaussian_kernel_normalised_and_symmetric():
    k = dsp.gaussian_kernel(5)
    assert k.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(k, k[::-1])


def test_gaussian_blur_preserves_mean_and_constants():
    data = np.full(50, 3.0)
    np.testing.assert_allclose(dsp.gaussian_blur(data), 3.0, atol=1e-9)
    rng = np.random.default_rng(3)
    noisy = rng.standard_normal(200)
    # circular blur conserves the total (sum), because the kernel sums to 1.
    assert dsp.gaussian_blur(noisy).sum() == pytest.approx(noisy.sum(), abs=1e-6)

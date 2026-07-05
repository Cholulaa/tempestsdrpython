"""Tests for IQ sources and the raw-file round trip."""

import numpy as np
import pytest

from tempestsdr.sources.file_source import FileSource, save_iq, _FORMATS


@pytest.mark.parametrize("fmt", ["float", "int8", "uint8", "int16", "uint16"])
def test_file_roundtrip(tmp_path, fmt):
    rng = np.random.default_rng(0)
    iq = (rng.uniform(-0.9, 0.9, 500) + 1j * rng.uniform(-0.9, 0.9, 500)).astype(np.complex64)
    path = tmp_path / f"cap_{fmt}.iq"
    save_iq(str(path), iq, sample_format=fmt)

    src = FileSource(str(path), samplerate=1e6, sample_format=fmt)
    read = src.read_all()
    assert read.size == iq.size
    # Quantisation tolerance depends on the format's bit depth.
    tol = 0.02 if fmt in ("int8", "uint8") else (1e-3 if "16" in fmt else 1e-6)
    np.testing.assert_allclose(read.real, iq.real, atol=tol)
    np.testing.assert_allclose(read.imag, iq.imag, atol=tol)


def test_file_blocks_and_max_samples(tmp_path):
    iq = np.arange(2000, dtype=np.float32) + 1j * np.arange(2000, dtype=np.float32)
    iq = iq.astype(np.complex64) / 4000.0
    path = tmp_path / "cap.iq"
    save_iq(str(path), iq, sample_format="float")

    src = FileSource(str(path), samplerate=1e6, sample_format="float",
                     block_size=256, max_samples=1000)
    blocks = list(src)
    assert sum(b.size for b in blocks) == 1000
    assert all(b.size <= 256 for b in blocks)


def test_file_loop(tmp_path):
    iq = (np.ones(100) + 1j * np.ones(100)).astype(np.complex64) * 0.5
    path = tmp_path / "cap.iq"
    save_iq(str(path), iq, sample_format="float")
    src = FileSource(str(path), samplerate=1e6, sample_format="float",
                     block_size=100, loop=True, max_samples=350)
    read = src.read_all()
    assert read.size == 350  # looped past the 100-sample file


def test_loop_on_empty_file_terminates(tmp_path):
    # Regression: loop=True on a file that yields zero complete samples must not
    # spin forever.
    empty = tmp_path / "empty.iq"
    empty.write_bytes(b"")
    src = FileSource(str(empty), samplerate=1e6, sample_format="uint8", loop=True)
    assert list(src) == []

    # A file with a single dangling byte (no complete I/Q pair) must also stop.
    odd = tmp_path / "odd.iq"
    odd.write_bytes(b"\x2a")
    src = FileSource(str(odd), samplerate=1e6, sample_format="uint8", loop=True)
    assert list(src) == []


def test_unknown_format_rejected(tmp_path):
    with pytest.raises(ValueError):
        FileSource("x.iq", samplerate=1e6, sample_format="nope")
    with pytest.raises(ValueError):
        save_iq(str(tmp_path / "x.iq"), np.zeros(4, np.complex64), sample_format="nope")


def test_uint8_is_rtlsdr_centred():
    # uint8 128 == 0.0 (rtl_sdr centres the ADC at 127.5/128).
    scale = _FORMATS["uint8"][1]
    np.testing.assert_allclose(scale(np.array([128], dtype=np.uint8)), [0.0])

"""Read raw interleaved I/Q recordings from disk.

Port of ``TSDRPlugin_RawFile`` (Martin Marinov, GPLv3).  Supported sample
formats and their normalisation match the C plugin exactly:

======  ==========  ===================================
name    dtype       scaling applied to reach ~[-1, 1]
======  ==========  ===================================
float   float32     none (already floating point)
int8    int8        value / 128
uint8   uint8       (value - 128) / 128   (rtl_sdr format)
int16   int16       value / 32767
uint16  uint16      (value - 32767) / 32767
======  ==========  ===================================

Samples are stored interleaved ``I0 Q0 I1 Q1 ...`` and folded into complex
values on read.
"""

from __future__ import annotations

import numpy as np

from .base import IQSource

_FORMATS = {
    "float": (np.float32, lambda a: a.astype(np.float32)),
    "float32": (np.float32, lambda a: a.astype(np.float32)),
    "int8": (np.int8, lambda a: a.astype(np.float32) / 128.0),
    "uint8": (np.uint8, lambda a: (a.astype(np.float32) - 128.0) / 128.0),
    "int16": (np.int16, lambda a: a.astype(np.float32) / 32767.0),
    "uint16": (np.uint16, lambda a: (a.astype(np.float32) - 32767.0) / 32767.0),
}

SAMPLES_TO_READ_AT_ONCE = 512 * 1024  # complex samples per block, as in the C plugin


class FileSource(IQSource):
    def __init__(
        self,
        path: str,
        samplerate: float,
        sample_format: str = "uint8",
        loop: bool = False,
        block_size: int = SAMPLES_TO_READ_AT_ONCE,
        max_samples: int | None = None,
    ) -> None:
        if sample_format not in _FORMATS:
            raise ValueError(
                f"unknown sample format {sample_format!r}; choose one of "
                f"{', '.join(sorted(_FORMATS))}"
            )
        self.path = path
        self.samplerate = float(samplerate)
        self.sample_format = sample_format
        self.loop = loop
        self.block_size = int(block_size)
        self.max_samples = max_samples
        self._dtype, self._scale = _FORMATS[sample_format]

    def __iter__(self):
        dtype = np.dtype(self._dtype)
        values_per_block = self.block_size * 2  # interleaved I/Q
        emitted = 0
        produced_this_pass = 0  # complex samples yielded since the last rewind
        with open(self.path, "rb") as fh:
            while True:
                raw = np.fromfile(fh, dtype=dtype, count=values_per_block)
                if raw.size % 2 != 0:
                    raw = raw[:-1]  # drop a dangling half-sample at EOF
                if raw.size == 0:
                    # End of file. Only rewind if this pass actually produced
                    # data, otherwise an empty/degenerate file would spin forever.
                    if self.loop and produced_this_pass > 0:
                        fh.seek(0)
                        produced_this_pass = 0
                        continue
                    break
                floats = self._scale(raw)
                iq = (floats[0::2] + 1j * floats[1::2]).astype(np.complex64)
                if self.max_samples is not None and emitted + iq.size > self.max_samples:
                    iq = iq[: self.max_samples - emitted]
                emitted += iq.size
                produced_this_pass += iq.size
                if iq.size:
                    yield iq
                if self.max_samples is not None and emitted >= self.max_samples:
                    break
                if raw.size < values_per_block and not self.loop:
                    break


def save_iq(path: str, iq: np.ndarray, sample_format: str = "uint8") -> None:
    """Write a complex array to disk in one of the supported raw formats.

    Inverse of :class:`FileSource`; used by the synthetic generator and tests.
    """
    if sample_format not in _FORMATS:
        raise ValueError(f"unknown sample format {sample_format!r}")
    iq = np.asarray(iq)
    inter = np.empty(iq.size * 2, dtype=np.float32)
    inter[0::2] = iq.real
    inter[1::2] = iq.imag
    if sample_format in ("float", "float32"):
        out = inter.astype(np.float32)
    elif sample_format == "int8":
        out = np.clip(np.round(inter * 128.0), -128, 127).astype(np.int8)
    elif sample_format == "uint8":
        out = np.clip(np.round(inter * 128.0 + 128.0), 0, 255).astype(np.uint8)
    elif sample_format == "int16":
        out = np.clip(np.round(inter * 32767.0), -32768, 32767).astype(np.int16)
    elif sample_format == "uint16":
        out = np.clip(np.round(inter * 32767.0 + 32767.0), 0, 65535).astype(np.uint16)
    out.tofile(path)

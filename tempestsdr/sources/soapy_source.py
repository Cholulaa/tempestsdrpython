"""SoapySDR live source (optional).

SoapySDR provides a single API over many front-ends (USRP/UHD, HackRF, Airspy,
LimeSDR, RTL-SDR, ...), so this one source covers most of the hardware the
original project supported through separate native plugins.  The import is
guarded; the toolkit runs fine without SoapySDR installed.
"""

from __future__ import annotations

import numpy as np

from .base import IQSource


class SoapySource(IQSource):
    def __init__(
        self,
        samplerate: float,
        center_freq: float,
        driver: str = "rtlsdr",
        gain: float | None = None,
        antenna: str | None = None,
        block_size: int = 256 * 1024,
        channel: int = 0,
    ) -> None:
        try:
            import SoapySDR
            from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
        except ImportError as exc:  # pragma: no cover - library dependent
            raise ImportError(
                "SoapySDR (with Python bindings) is required for this source. "
                "See https://github.com/pothosware/SoapySDR."
            ) from exc

        self._SoapySDR = SoapySDR
        self._SOAPY_SDR_RX = SOAPY_SDR_RX
        self._SOAPY_SDR_CF32 = SOAPY_SDR_CF32

        self._sdr = SoapySDR.Device({"driver": driver})
        self._channel = int(channel)
        self._sdr.setSampleRate(SOAPY_SDR_RX, channel, float(samplerate))
        self._sdr.setFrequency(SOAPY_SDR_RX, channel, float(center_freq))
        if gain is not None:
            self._sdr.setGain(SOAPY_SDR_RX, channel, float(gain))
        if antenna is not None:
            self._sdr.setAntenna(SOAPY_SDR_RX, channel, antenna)

        self.samplerate = float(samplerate)
        self.center_freq = float(center_freq)
        self.block_size = int(block_size)
        self._running = False
        self._stream = None

    def __iter__(self):
        self._stream = self._sdr.setupStream(
            self._SOAPY_SDR_RX, self._SOAPY_SDR_CF32, [self._channel]
        )
        self._sdr.activateStream(self._stream)
        self._running = True
        buff = np.empty(self.block_size, dtype=np.complex64)
        try:
            while self._running:
                sr = self._sdr.readStream(self._stream, [buff], self.block_size)
                n = sr.ret
                if n > 0:
                    yield buff[:n].copy()
        finally:
            self._sdr.deactivateStream(self._stream)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:  # pragma: no cover - hardware dependent
        try:
            if self._stream is not None:
                self._sdr.closeStream(self._stream)
        except Exception:
            pass

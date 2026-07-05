"""RTL-SDR live source (optional).

Requires the ``pyrtlsdr`` package and an RTL2832U dongle.  The import is guarded
so the rest of the toolkit works without the hardware or the library installed.
This is the Python equivalent of a native RTL-SDR TSDR plugin.
"""

from __future__ import annotations

import numpy as np

from .base import IQSource


class RtlSdrSource(IQSource):
    def __init__(
        self,
        samplerate: float,
        center_freq: float,
        gain: float | str = "auto",
        block_size: int = 256 * 1024,
        device_index: int = 0,
    ) -> None:
        try:
            from rtlsdr import RtlSdr
        except ImportError as exc:  # pragma: no cover - hardware/library dependent
            raise ImportError(
                "pyrtlsdr is required for the RTL-SDR source. Install it with "
                "'pip install pyrtlsdr' and make sure the librtlsdr driver is "
                "present."
            ) from exc
        except AttributeError as exc:  # pragma: no cover - env dependent
            # e.g. "function 'rtlsdr_set_dithering' not found": pyrtlsdr is newer
            # than the installed librtlsdr.dll.
            raise RuntimeError(
                f"pyrtlsdr could not bind to your librtlsdr ({exc}). The DLL is "
                "older than pyrtlsdr expects. Either downgrade pyrtlsdr "
                "(pip install 'pyrtlsdr<0.3') or supply a newer rtlsdr.dll."
            ) from exc

        self._sdr = RtlSdr(device_index=device_index)
        self._sdr.sample_rate = float(samplerate)
        self._sdr.center_freq = float(center_freq)
        self._apply_gain(gain)
        self.samplerate = float(samplerate)
        self.center_freq = float(center_freq)
        self.block_size = int(block_size)
        self._running = False

    def _apply_gain(self, gain) -> None:
        # pyrtlsdr's gain setter wants a number; "auto" means enable tuner AGC.
        if gain in ("auto", None, ""):
            self._sdr.set_manual_gain_enabled(False)
        else:
            self._sdr.gain = float(gain)

    def set_center_freq(self, freq: float) -> None:
        self.center_freq = float(freq)
        self._sdr.center_freq = float(freq)

    def set_gain(self, gain: float | str) -> None:
        self._apply_gain(gain)

    def __iter__(self):
        self._running = True
        while self._running:
            samples = self._sdr.read_samples(self.block_size)
            yield np.asarray(samples, dtype=np.complex64)

    def stop(self) -> None:
        self._running = False

    def close(self) -> None:  # pragma: no cover - hardware dependent
        try:
            self._sdr.close()
        except Exception:
            pass

"""Abstract IQ sample source.

Equivalent of the plugin interface in ``TempestSDR/src/include/TSDRPlugin.h``.
A source produces blocks of complex baseband samples; the processor consumes
them.  Concrete sources: :class:`~tempestsdr.sources.file_source.FileSource`
(the ``TSDRPlugin_RawFile`` equivalent) and the optional hardware sources.
"""

from __future__ import annotations

import abc

import numpy as np


class IQSource(abc.ABC):
    """Base class for anything that yields complex baseband blocks."""

    samplerate: float

    @abc.abstractmethod
    def __iter__(self):
        """Yield ``numpy.ndarray`` blocks of complex samples."""

    def read_all(self) -> np.ndarray:
        """Concatenate every block into one array (offline convenience)."""
        blocks = list(self)
        if not blocks:
            return np.zeros(0, dtype=np.complex64)
        return np.concatenate(blocks)

    def close(self) -> None:  # pragma: no cover - default no-op
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

"""TempestSDR — Python port.

A software toolkit for reconstructing the image on a video display from its
unintended electromagnetic emanations, captured with a software-defined radio.
This is a from-scratch Python re-implementation of Martin Marinov's TempestSDR
(https://github.com/martinmarinov/TempestSDR), released under the same GPLv3
licence.

Intended for security research, education and defensive evaluation (e.g.
measuring how much a given setup leaks and testing shielding).  See the README
for the responsible-use note.

Typical use::

    from tempestsdr import TempestProcessor, ProcessorConfig
    from tempestsdr.sources import FileSource

    src = FileSource("capture.iq", samplerate=8_000_000, sample_format="uint8")
    proc = TempestProcessor(ProcessorConfig(
        samplerate=src.samplerate, height=628, refresh_rate=60))
    frames = proc.reconstruct(src.read_all())
"""

from .processor import ProcessorConfig, TempestProcessor
from .framerate import FrameRateDetector
from .sync import SyncDetector
from . import dsp, videomodes, synth

__version__ = "1.0.0"

__all__ = [
    "TempestProcessor",
    "ProcessorConfig",
    "FrameRateDetector",
    "SyncDetector",
    "dsp",
    "videomodes",
    "synth",
    "__version__",
]

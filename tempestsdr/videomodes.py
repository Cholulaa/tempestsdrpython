"""VESA video-mode presets.

Port of ``JavaGUI/src/martin/tempest/gui/VideoMode.java`` (Martin Marinov,
GPLv3).  ``width`` and ``height`` are the *total* pixel/line counts including
blanking (i.e. the pixel clock geometry), not the visible resolution named in
the label.  ``refresh_rate`` is the vertical refresh in Hz.

Only ``height`` and ``refresh_rate`` are needed to drive the reconstruction:
the receiver sample rate together with these two values determines the pixel
geometry (see :mod:`tempestsdr.processor`).  ``width`` is retained for
reference and for the synthetic signal generator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoMode:
    name: str
    width: int          # total pixels per line (incl. horizontal blanking)
    height: int         # total lines per frame (incl. vertical blanking)
    refresh_rate: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.name


# Order and values copied verbatim from VideoMode.java's MODES table.
MODES: tuple[VideoMode, ...] = (
    VideoMode("PAL TV", 576, 625, 25),
    VideoMode("640x400 @ 85Hz", 832, 445, 85),
    VideoMode("720x400 @ 85Hz", 936, 446, 85),
    VideoMode("640x480 @ 60Hz", 800, 525, 60),
    VideoMode("640x480 @ 100Hz", 848, 509, 100),
    VideoMode("640x480 @ 72Hz", 832, 520, 72),
    VideoMode("640x480 @ 75Hz", 840, 500, 75),
    VideoMode("640x480 @ 85Hz", 832, 509, 85),
    VideoMode("768x576 @ 60 Hz", 976, 597, 60),
    VideoMode("768x576 @ 72 Hz", 992, 601, 72),
    VideoMode("768x576 @ 75 Hz", 1008, 602, 75),
    VideoMode("768x576 @ 85 Hz", 1008, 605, 85),
    VideoMode("768x576 @ 100 Hz", 1024, 611, 100),
    VideoMode("800x600 @ 56Hz", 1024, 625, 56),
    VideoMode("800x600 @ 60Hz", 1056, 628, 60),
    VideoMode("800x600 @ 72Hz", 1040, 666, 72),
    VideoMode("800x600 @ 75Hz", 1056, 625, 75),
    VideoMode("800x600 @ 85Hz", 1048, 631, 85),
    VideoMode("800x600 @ 100Hz", 1072, 636, 100),
    VideoMode("1024x600 @ 60 Hz", 1312, 622, 60),
    VideoMode("1024x768i @ 43Hz", 1264, 817, 43),
    VideoMode("1024x768 @ 60Hz", 1344, 806, 60),
    VideoMode("1024x768 @ 70Hz", 1328, 806, 70),
    VideoMode("1024x768 @ 75Hz", 1312, 800, 75),
    VideoMode("1024x768 @ 85Hz", 1376, 808, 85),
    VideoMode("1024x768 @ 100Hz", 1392, 814, 100),
    VideoMode("1024x768 @ 120Hz", 1408, 823, 120),
    VideoMode("1152x864 @ 60Hz", 1520, 895, 60),
    VideoMode("1152x864 @ 75Hz", 1600, 900, 75),
    VideoMode("1152x864 @ 85Hz", 1552, 907, 85),
    VideoMode("1152x864 @ 100Hz", 1568, 915, 100),
    VideoMode("1280x768 @ 60 Hz", 1680, 795, 60),
    VideoMode("1280x800 @ 60 Hz", 1680, 828, 60),
    VideoMode("1280x960 @ 60Hz", 1800, 1000, 60),
    VideoMode("1280x960 @ 75Hz", 1728, 1002, 75),
    VideoMode("1280x960 @ 85Hz", 1728, 1011, 85),
    VideoMode("1280x960 @ 100Hz", 1760, 1017, 100),
    VideoMode("1280x1024 @ 60Hz", 1688, 1066, 60),
    VideoMode("1280x1024 @ 75Hz", 1688, 1066, 75),
    VideoMode("1280x1024 @ 85Hz", 1728, 1072, 85),
    VideoMode("1280x1024 @ 100Hz", 1760, 1085, 100),
    VideoMode("1280x1024 @ 120Hz", 1776, 1097, 120),
    VideoMode("1368x768 @ 60 Hz", 1800, 795, 60),
    VideoMode("1400x1050 @ 60Hz", 1880, 1082, 60),
    VideoMode("1400x1050 @ 72 Hz", 1896, 1094, 72),
    VideoMode("1400x1050 @ 75 Hz", 1896, 1096, 75),
    VideoMode("1400x1050 @ 85 Hz", 1912, 1103, 85),
    VideoMode("1400x1050 @ 100 Hz", 1928, 1112, 100),
    VideoMode("1440x900 @ 60 Hz", 1904, 932, 60),
    VideoMode("1440x1050 @ 60 Hz", 1936, 1087, 60),
    VideoMode("1600x1000 @ 60Hz", 2144, 1035, 60),
    VideoMode("1600x1000 @ 75Hz", 2160, 1044, 75),
    VideoMode("1600x1000 @ 85Hz", 2176, 1050, 85),
    VideoMode("1600x1000 @ 100Hz", 2192, 1059, 100),
    VideoMode("1600x1024 @ 60Hz", 2144, 1060, 60),
    VideoMode("1600x1024 @ 75Hz", 2176, 1069, 75),
    VideoMode("1600x1024 @ 76Hz", 2096, 1070, 76),
    VideoMode("1600x1024 @ 85Hz", 2176, 1075, 85),
    VideoMode("1600x1200 @ 60Hz", 2160, 1250, 60),
    VideoMode("1600x1200 @ 65Hz", 2160, 1250, 65),
    VideoMode("1600x1200 @ 70Hz", 2160, 1250, 70),
    VideoMode("1600x1200 @ 75Hz", 2160, 1250, 75),
    VideoMode("1600x1200 @ 85Hz", 2160, 1250, 85),
    VideoMode("1600x1200 @ 100 Hz", 2208, 1271, 100),
    VideoMode("1680x1050 @ 60Hz (reduced blanking)", 1840, 1080, 60),
    VideoMode("1680x1050 @ 60Hz (non-interlaced)", 2240, 1089, 60),
    VideoMode("1680x1050 @ 60 Hz", 2256, 1087, 60),
    VideoMode("1792x1344 @ 60Hz", 2448, 1394, 60),
    VideoMode("1792x1344 @ 75Hz", 2456, 1417, 75),
    VideoMode("1856x1392 @ 60Hz", 2528, 1439, 60),
    VideoMode("1856x1392 @ 75Hz", 2560, 1500, 75),
    VideoMode("1920x1080 @ 60Hz", 2576, 1125, 60),
    VideoMode("1920x1080 @ 75Hz", 2608, 1126, 75),
    VideoMode("1920x1200 @ 60Hz", 2592, 1242, 60),
    VideoMode("1920x1200 @ 75Hz", 2624, 1253, 75),
    VideoMode("1920x1440 @ 60Hz", 2600, 1500, 60),
    VideoMode("1920x1440 @ 75Hz", 2640, 1500, 75),
    VideoMode("1920x2400 @ 25Hz", 2048, 2434, 25),
    VideoMode("1920x2400 @ 30Hz", 2044, 2434, 30),
    VideoMode("2048x1536 @ 60Hz", 2800, 1589, 60),
)


def get_video_modes() -> tuple[VideoMode, ...]:
    return MODES


def find_by_name(name: str) -> VideoMode | None:
    for mode in MODES:
        if mode.name == name:
            return mode
    return None


def find_closest(refresh_rate: float, height: int) -> VideoMode | None:
    """Nearest preset for a measured (refresh_rate, total-height) pair.

    Mirrors ``VideoMode.findClosestVideoModeId(framerate, height, modes)``:
    prefer an exact height match with the closest refresh rate, otherwise fall
    back to the closest total-height.
    """
    if not MODES:
        return None
    best = None
    best_diff = float("inf")
    for mode in MODES:
        if mode.height == height:
            delta = abs(mode.refresh_rate - refresh_rate)
            if delta < best_diff:
                best_diff = delta
                best = mode
    if best is not None:
        return best
    best_idiff = float("inf")
    for mode in MODES:
        delta = abs(mode.height - height)
        if delta < best_idiff:
            best_idiff = delta
            best = mode
    return best

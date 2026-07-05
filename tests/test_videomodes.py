"""Tests for the VESA video-mode preset table."""

from tempestsdr import videomodes


def test_modes_present_and_wellformed():
    modes = videomodes.get_video_modes()
    assert len(modes) > 50
    for m in modes:
        assert m.width > 0 and m.height > 0 and m.refresh_rate > 0


def test_known_modes():
    m = videomodes.find_by_name("640x480 @ 60Hz")
    assert m is not None
    assert (m.width, m.height, m.refresh_rate) == (800, 525, 60)

    m = videomodes.find_by_name("1920x1080 @ 60Hz")
    assert (m.width, m.height, m.refresh_rate) == (2576, 1125, 60)


def test_find_by_name_missing():
    assert videomodes.find_by_name("does not exist") is None


def test_find_closest_exact_height():
    # height 525 matches "640x480 @ 60Hz"; nearest refresh picks it.
    m = videomodes.find_closest(refresh_rate=61.0, height=525)
    assert m.height == 525


def test_find_closest_falls_back_to_nearest_height():
    m = videomodes.find_closest(refresh_rate=60.0, height=9999)
    assert m is not None  # returns the closest available height

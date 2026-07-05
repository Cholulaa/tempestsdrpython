"""Tests for the edge agent (payload building + command handling).

Network I/O is stubbed, so these don't need a running server; the full
agent<->server round trip is covered by the deploy README's no-hardware recipe.
"""

import base64
import time

import pytest

pytest.importorskip("PIL")  # the engine encodes JPEG frames

from tempestsdr.agent import Agent


@pytest.fixture(scope="module")
def agent():
    a = Agent("http://example.invalid", "probe-x", send_interval=1.0)
    a.use_synthetic()
    time.sleep(1.5)  # let the engine reconstruct a frame
    yield a
    a.engine.stop()


def test_metadata_shape(agent):
    m = agent._metadata()
    assert m["device_id"] == "probe-x"
    assert "geometry" in m and "params" in m and "snr" in m
    assert "autocorr" not in m  # stripped to keep the uplink small


def test_check_in_builds_payload_and_returns_commands(agent):
    captured = {}

    def fake_post(payload):
        captured["payload"] = payload
        return {"ok": True, "commands": [{"type": "detect"}]}

    agent._post = fake_post
    cmds = agent.check_in()
    assert cmds == [{"type": "detect"}]
    p = captured["payload"]
    assert p["device_id"] == "probe-x"
    assert p["meta"]["device_id"] == "probe-x"
    # a real JPEG frame was base64-encoded
    assert base64.b64decode(p["frame_b64"]).startswith(b"\xff\xd8")


def test_apply_command_config_and_mode(agent):
    agent._apply_command({"type": "set_config", "motion_blur": 0.9, "nearest": True})
    st = agent.engine.status()
    assert st["params"]["motion_blur"] == pytest.approx(0.9)
    assert st["params"]["nearest"] is True

    agent._apply_command({"type": "set_mode", "height": 806, "refresh_rate": 70.0})
    st = agent.engine.status()
    assert st["geometry"]["height"] == 806
    assert st["geometry"]["refresh_rate"] == pytest.approx(70.0)


def test_apply_command_nudge(agent):
    agent._apply_command({"type": "nudge", "direction": "down", "pixels": 3})
    assert agent.engine.status()["manual"]["dy"] == 3
    agent._apply_command({"type": "nudge", "direction": "reset"})
    assert agent.engine.status()["manual"] == {"dx": 0, "dy": 0}


def test_apply_command_unknown_is_ignored(agent):
    # must not raise
    agent._apply_command({"type": "totally-unknown"})

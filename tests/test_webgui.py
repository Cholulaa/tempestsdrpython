"""Smoke tests for the web control panel (skipped if Flask is unavailable)."""

import time

import numpy as np
import pytest

flask = pytest.importorskip("flask")
PIL = pytest.importorskip("PIL")

from tempestsdr.webgui import Engine, LoopArraySource, create_app


@pytest.fixture(scope="module")
def engine():
    eng = Engine()
    # give the worker a moment to reconstruct at least one frame
    for _ in range(50):
        if eng.status()["fps"] > 0 or eng.latest_jpeg():
            break
        time.sleep(0.05)
    yield eng
    eng.stop()


def test_engine_produces_frames_and_status(engine):
    time.sleep(1.0)
    st = engine.status()
    assert st["running"] is True
    assert st["geometry"]["width"] > 0
    assert engine.latest_jpeg().startswith(b"\xff\xd8")  # JPEG magic


def test_loop_source_wraps():
    src = LoopArraySource(np.arange(10, dtype=np.complex64), samplerate=1e6, block=4)
    it = iter(src)
    blocks = [next(it) for _ in range(4)]
    src.stop()
    assert [b.size for b in blocks] == [4, 4, 2, 4]  # wrapped back to the start


def test_reconfigure_and_nudge(engine):
    engine.reconfigure(motion_blur=0.8, nearest=True, autoshift=False)
    st = engine.status()
    assert st["params"]["motion_blur"] == pytest.approx(0.8)
    assert st["params"]["nearest"] is True
    engine.nudge("down", 4)
    assert engine.status()["manual"]["dy"] == 4
    engine.nudge("reset")
    assert engine.status()["manual"] == {"dx": 0, "dy": 0}


def test_flask_routes(engine):
    app = create_app(engine)
    client = app.test_client()
    assert client.get("/").status_code == 200
    assert client.get("/api/status").status_code == 200
    modes = client.get("/api/modes").get_json()
    assert isinstance(modes, list) and len(modes) > 10
    r = client.post("/api/config", json={"motion_blur": 0.3})
    assert r.status_code == 200
    assert client.get("/snapshot.png").data.startswith(b"\x89PNG")

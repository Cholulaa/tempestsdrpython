"""Tests for the control server (Fleet registry + Flask routes)."""

import base64

import pytest

flask = pytest.importorskip("flask")

from tempestsdr.server import Fleet, create_app


def test_fleet_ingest_and_commands():
    fleet = Fleet()
    # unknown device with a queued command
    fleet.queue_command("p1", {"type": "detect"})
    cmds = fleet.ingest("p1", {"snr": 2.0}, b"\xff\xd8jpeg")
    assert cmds == [{"type": "detect"}]
    # command queue is drained after delivery
    assert fleet.ingest("p1", {}, None) == []
    devs = fleet.devices()
    assert len(devs) == 1 and devs[0]["id"] == "p1" and devs[0]["has_frame"]
    assert fleet.frame("p1") == b"\xff\xd8jpeg"


def test_fleet_archive(tmp_path):
    fleet = Fleet(archive_dir=str(tmp_path))
    fleet.ingest("cam-1", {}, b"\xff\xd8data")
    files = list(tmp_path.rglob("*.jpg"))
    assert len(files) == 1 and files[0].read_bytes() == b"\xff\xd8data"


@pytest.fixture
def client():
    return create_app(Fleet()).test_client()


def test_ingest_route_roundtrip(client):
    frame = base64.b64encode(b"\xff\xd8xy").decode()
    r = client.post("/api/ingest", json={"device_id": "d1", "meta": {"snr": 1},
                                         "frame_b64": frame})
    assert r.status_code == 200 and r.get_json()["ok"]
    assert client.get("/api/devices").get_json()[0]["id"] == "d1"
    assert client.get("/api/frame/d1").data == b"\xff\xd8xy"


def test_ingest_requires_device_id(client):
    assert client.post("/api/ingest", json={}).status_code == 400


def test_command_route_queues(client):
    client.post("/api/ingest", json={"device_id": "d1", "meta": {}})
    client.post("/api/command/d1", json={"type": "set_freq", "freq": 4e8})
    # command is delivered on the next ingest
    cmds = client.post("/api/ingest", json={"device_id": "d1"}).get_json()["commands"]
    assert cmds == [{"type": "set_freq", "freq": 4e8}]


def test_command_requires_type(client):
    assert client.post("/api/command/d1", json={}).status_code == 400


def test_missing_frame_404(client):
    assert client.get("/api/frame/nope").status_code == 404


def test_api_key_enforced():
    client = create_app(Fleet(api_key="secret")).test_client()
    assert client.post("/api/ingest", json={"device_id": "d"}).status_code == 401
    ok = client.post("/api/ingest", json={"device_id": "d"},
                     headers={"X-API-Key": "secret"})
    assert ok.status_code == 200

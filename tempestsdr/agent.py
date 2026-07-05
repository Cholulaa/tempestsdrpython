"""Autonomous edge probe: reconstruct locally, report to a control server.

Designed to run headless on a small board (e.g. a Raspberry Pi with an RTL-SDR):
it runs the reconstruction pipeline continuously and, every ``send_interval``
seconds, POSTs the latest recovered frame plus a little metadata to a
:mod:`tempestsdr.server` control server.  The server's response carries any
queued commands (retune, change resolution, auto-detect, switch source), which
the probe applies live.

The uplink uses only the Python standard library (``urllib``) so the probe stays
dependency-light; it keeps running through network outages and reconnects with
backoff.  A ``--synthetic`` mode lets the whole loop be exercised with no radio
hardware.

Deploy only against equipment you own or are explicitly authorised to test.
"""

from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request


class Agent:
    def __init__(self, server_url: str, device_id: str, *, api_key: str | None = None,
                 send_interval: float = 2.0, jpeg_quality: int = 75):
        from .webgui import Engine  # reuse the pipeline runner (no Flask needed)
        self.server_url = server_url.rstrip("/")
        self.device_id = device_id
        self.api_key = api_key
        self.send_interval = float(send_interval)
        self.jpeg_quality = int(jpeg_quality)
        self.engine = Engine()          # starts on the synthetic source by default
        self.frequency = None
        self.gain = "auto"
        self._running = False

    # -- source setup ------------------------------------------------------
    def use_synthetic(self):
        self.engine.start_synthetic()

    def use_sdr(self, driver, samplerate, frequency, gain="auto", height=None, refresh=None):
        self.frequency, self.gain = float(frequency), gain
        self.engine.start_sdr(driver, samplerate, frequency, gain)
        if height and refresh:
            self.engine.reconfigure(height=int(height), refresh_rate=float(refresh))

    def use_file(self, path, samplerate, sample_format="uint8"):
        self.engine.start_file(path, samplerate, sample_format)

    # -- command handling --------------------------------------------------
    def _apply_command(self, cmd: dict) -> None:
        t = cmd.get("type")
        try:
            if t == "set_freq":
                self.frequency = float(cmd["freq"])
                self.engine.set_frequency(self.frequency)
            elif t == "set_gain":
                self.gain = cmd["gain"]
                self.engine.set_gain(self.gain)
            elif t == "set_mode":
                self.engine.reconfigure(height=int(cmd["height"]),
                                        refresh_rate=float(cmd["refresh_rate"]))
            elif t == "set_config":
                self.engine.reconfigure(**{k: v for k, v in cmd.items() if k != "type"})
            elif t == "detect":
                self.engine.apply_detection()
            elif t == "nudge":
                self.engine.nudge(cmd.get("direction", "reset"), int(cmd.get("pixels", 4)))
            elif t == "set_source":
                kind = cmd.get("kind")
                if kind == "synthetic":
                    self.use_synthetic()
                elif kind == "sdr":
                    self.use_sdr(cmd.get("driver", "rtlsdr"), float(cmd["samplerate"]),
                                 float(cmd["frequency"]), cmd.get("gain", "auto"))
                elif kind == "file":
                    self.use_file(cmd["path"], float(cmd["samplerate"]),
                                  cmd.get("format", "uint8"))
        except Exception as exc:
            print(f"[agent] command {t} failed: {exc}")

    # -- uplink ------------------------------------------------------------
    def _metadata(self) -> dict:
        st = self.engine.status()
        st.pop("autocorr", None)  # keep the payload small
        st["freq"] = self.frequency
        st["gain"] = self.gain
        st["device_id"] = self.device_id
        return st

    def _post(self, payload: dict) -> dict | None:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.server_url}/api/ingest", data=body, method="POST",
            headers={"Content-Type": "application/json",
                     **({"X-API-Key": self.api_key} if self.api_key else {})})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode())

    def check_in(self) -> list[dict]:
        """Send the latest frame + metadata, return commands from the server."""
        frame = self.engine.latest_jpeg()
        payload = {
            "device_id": self.device_id,
            "meta": self._metadata(),
            "frame_b64": base64.b64encode(frame).decode() if frame else None,
        }
        resp = self._post(payload)
        return (resp or {}).get("commands", [])

    def run(self):
        """Run the report/command loop until stopped, tolerating outages."""
        self._running = True
        backoff = self.send_interval
        print(f"[agent] {self.device_id} -> {self.server_url} "
              f"(every {self.send_interval}s)")
        while self._running:
            try:
                for cmd in self.check_in():
                    self._apply_command(cmd)
                backoff = self.send_interval
                time.sleep(self.send_interval)
            except (urllib.error.URLError, OSError) as exc:
                # Network down: keep reconstructing locally, retry with backoff.
                print(f"[agent] uplink failed ({exc}); retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            except KeyboardInterrupt:
                break
        self._running = False
        self.engine.stop()

    def stop(self):
        self._running = False

"""Full web control panel for TempestSDR.

A browser-based front-end that goes beyond the original Java GUI: it runs the
reconstruction pipeline in a background worker and streams the recovered image
live (MJPEG), while exposing every processor knob, a blind auto-detect button, a
clickable frame-rate autocorrelation plot, manual sync nudges, and three source
types — a built-in synthetic emanation (so it works with no hardware), an
uploaded raw-IQ capture, or a live SDR.

Launch with ``tempestsdr webgui`` and open http://127.0.0.1:8000.

Only :func:`run` pulls in Flask/Pillow, so importing the rest of the package
never requires them.
"""

from __future__ import annotations

import io
import threading
import time
from collections import deque

import numpy as np

from . import videomodes
from .dsp import am_demodulate
from .processor import ProcessorConfig, TempestProcessor

# Geometry of the built-in synthetic target (a real VESA-ish mode, kept modest
# so reconstruction is responsive).  samplerate is chosen so the processor's
# computed width equals total_width, i.e. a clean 1:1 default reconstruction.
_DEMO_TOTAL_W = 1056
_DEMO_TOTAL_H = 628          # 800x600 @ 60 Hz total geometry (in the detector band)
_DEMO_REFRESH = 60.0
_DEMO_SAMPLERATE = _DEMO_TOTAL_W * _DEMO_TOTAL_H * _DEMO_REFRESH / 2  # ~19.9 Msps


def _make_test_image(h: int = 300, w: int = 400) -> np.ndarray:
    from PIL import Image, ImageDraw

    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rectangle([15, 12, w - 15, 46], fill=255)
    d.text((24, 20), "TEMPEST SDR - control panel", fill=0)
    d.text((24, 70), "reconstructed live from", fill=210)
    d.text((24, 96), "compromising emanations", fill=230)
    d.text((24, 140), "no radio hardware required", fill=180)
    d.rectangle([24, 175, w - 24, 215], outline=255, width=2)
    for i in range(w - 60):
        im.putpixel((24 + i, 240), int(255 * i / (w - 60)))
    return np.asarray(im, dtype=np.float64)


def _encode_jpeg(frame01: np.ndarray, quality: int = 80) -> bytes:
    from PIL import Image

    arr = (np.clip(frame01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class LoopArraySource:
    """Yield an in-memory IQ array in fixed blocks, looping forever."""

    def __init__(self, iq: np.ndarray, samplerate: float, block: int = 65536):
        self.iq = iq
        self.samplerate = float(samplerate)
        self.block = int(block)
        self._running = True

    def __iter__(self):
        i = 0
        n = self.iq.size
        while self._running:
            if i >= n:
                i = 0
            yield self.iq[i:i + self.block]
            i += self.block

    def stop(self):
        self._running = False


class Engine:
    """Owns the processor, the active source and the worker thread."""

    def __init__(self):
        self.lock = threading.RLock()
        self._generation = 0
        self._thread = None
        self._source = None
        self.source_kind = "synthetic"
        self.true_geometry = None  # ground-truth mode of the synthetic source
        self._frame_times = deque(maxlen=30)
        self._latest_jpeg = _encode_jpeg(np.zeros((10, 10)))
        self._manual_dx = 0
        self._manual_dy = 0
        self._autocorr = None
        self._last_autocorr_t = 0.0

        cfg = ProcessorConfig(
            samplerate=_DEMO_SAMPLERATE, height=_DEMO_TOTAL_H,
            refresh_rate=_DEMO_REFRESH, motion_blur=0.5, detect_framerate=True,
        )
        self.processor = TempestProcessor(cfg)
        self.start_synthetic()

    # -- sources -----------------------------------------------------------
    def _build_synthetic_iq(self):
        from .synth import SyntheticConfig, generate
        img = _make_test_image()
        scfg = SyntheticConfig(
            total_width=_DEMO_TOTAL_W, total_height=_DEMO_TOTAL_H,
            refresh_rate=_DEMO_REFRESH, samplerate=_DEMO_SAMPLERATE,
            num_frames=8, snr_db=12.0, start_offset_pixels=4000, seed=1,
        )
        return generate(img, scfg)

    def start_synthetic(self):
        iq = self._build_synthetic_iq()
        self.true_geometry = {
            "height": _DEMO_TOTAL_H, "refresh_rate": _DEMO_REFRESH,
            "samplerate": _DEMO_SAMPLERATE,
        }
        with self.lock:
            self.processor.reconfigure(
                samplerate=_DEMO_SAMPLERATE, height=_DEMO_TOTAL_H,
                refresh_rate=_DEMO_REFRESH)
        self._start_source("synthetic", LoopArraySource(iq, _DEMO_SAMPLERATE))

    def start_file(self, path, samplerate, sample_format="uint8"):
        from .sources.file_source import FileSource
        src = FileSource(path, samplerate=samplerate, sample_format=sample_format, loop=True)
        with self.lock:
            self.processor.reconfigure(samplerate=float(samplerate))
        self.true_geometry = None
        self._start_source("file", src)

    def start_sdr(self, driver, samplerate, frequency, gain="auto"):
        if driver == "rtlsdr-native":
            from .sources.rtlsdr_source import RtlSdrSource
            src = RtlSdrSource(samplerate, frequency, gain=gain)
        else:
            from .sources.soapy_source import SoapySource
            src = SoapySource(samplerate, frequency, driver=driver,
                              gain=None if gain in ("auto", None) else float(gain))
        with self.lock:
            self.processor.reconfigure(samplerate=float(samplerate))
        self.true_geometry = None
        self._start_source("sdr", src)

    def _start_source(self, kind, source):
        self.stop()
        with self.lock:
            self._generation += 1
            gen = self._generation
            self._source = source
            self.source_kind = kind
            self.processor.reset()
        self._thread = threading.Thread(target=self._worker, args=(source, gen), daemon=True)
        self._thread.start()

    def stop(self):
        src = self._source
        if src is not None and hasattr(src, "stop"):
            try:
                src.stop()
            except Exception:
                pass
        self._source = None

    # -- worker ------------------------------------------------------------
    def _worker(self, source, gen):
        try:
            for block in source:
                if gen != self._generation:
                    break
                with self.lock:
                    frames = self.processor.process(block)
                for frame in frames:
                    self._on_frame(frame)
                if not frames:
                    time.sleep(0.001)
        except Exception as exc:  # keep the server alive on source errors
            self._autocorr = None
            print(f"[webgui] source stopped: {exc}")

    def _on_frame(self, frame):
        if self._manual_dx or self._manual_dy:
            frame = np.roll(frame, shift=(self._manual_dy, self._manual_dx), axis=(0, 1))
        self._latest_jpeg = _encode_jpeg(frame)
        self._frame_times.append(time.time())
        now = time.time()
        if now - self._last_autocorr_t > 0.4:
            self._last_autocorr_t = now
            self._update_autocorr()

    def _update_autocorr(self):
        det = self.processor.frame_detector
        if det is None:
            return
        corr = det.correlation
        if corr.size == 0:
            return
        sr = self.processor.config.samplerate
        lo = int(sr / det.max_framerate)
        hi = min(int(sr / det.min_framerate), corr.size - 1)
        if hi <= lo:
            return
        band = corr[lo:hi]
        # Downsample to at most 300 points for the browser plot.
        step = max(1, band.size // 300)
        vals = band[::step]
        vmin, vmax = float(vals.min()), float(vals.max())
        norm = (vals - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(vals)
        peak_rel = int(np.argmax(band))
        peak_lag = lo + peak_rel
        self._autocorr = {
            "values": [round(float(v), 4) for v in norm],
            "lag_lo": lo, "lag_hi": hi, "step": step,
            "peak_lag": peak_lag, "peak_refresh": sr / peak_lag,
        }

    # -- controls ----------------------------------------------------------
    def reconfigure(self, **changes):
        clean = {}
        for k, v in changes.items():
            if k in ("height",):
                clean[k] = int(v)
            elif k in ("samplerate", "refresh_rate", "motion_blur", "norm_coeff"):
                clean[k] = float(v)
            elif k in ("autoshift", "nearest", "lowpass_before_sync",
                       "autogain_after", "framerate_pll", "detect_framerate"):
                clean[k] = bool(v)
        with self.lock:
            self.processor.reconfigure(**clean)

    def nudge(self, direction, pixels=1):
        w, h = self.processor.width, self.processor.config.height
        if direction == "up":
            self._manual_dy -= pixels
        elif direction == "down":
            self._manual_dy += pixels
        elif direction == "left":
            self._manual_dx -= pixels
        elif direction == "right":
            self._manual_dx += pixels
        elif direction == "reset":
            self._manual_dx = self._manual_dy = 0
        self._manual_dx %= max(1, w)
        self._manual_dy %= max(1, h)

    def detect(self):
        with self.lock:
            det = self.processor.frame_detector
            if det is None:
                return None
            est = det.estimate_resolution()
        return est

    def apply_detection(self):
        est = self.detect()
        if not est:
            return None
        height = int(round(est["height_lines"]))
        refresh = est["refresh_rate"]
        self.reconfigure(height=height, refresh_rate=refresh)
        return {"height": height, "refresh_rate": refresh}

    def status(self):
        p = self.processor
        times = list(self._frame_times)
        fps = 0.0
        if len(times) >= 2 and times[-1] > times[0]:
            fps = (len(times) - 1) / (times[-1] - times[0])
        return {
            "running": self._source is not None,
            "source_kind": self.source_kind,
            "true_geometry": self.true_geometry,
            "geometry": {
                "width": p.width, "height": p.config.height,
                "refresh_rate": round(p.config.refresh_rate, 4),
                "samplerate": p.config.samplerate,
                "pixels_per_sample": round(p.pixels_per_sample, 4),
            },
            "params": {
                "motion_blur": round(p.config.motion_blur, 3),
                "nearest": p.config.nearest,
                "autoshift": p.config.autoshift,
                "autogain_after": p.config.autogain_after,
                "lowpass_before_sync": p.config.lowpass_before_sync,
                "framerate_pll": p.config.framerate_pll,
            },
            "manual": {"dx": self._manual_dx, "dy": self._manual_dy},
            "snr": round(float(p._autogain.snr), 3),
            "locked": bool(p._sync.locked),
            "fps": round(fps, 1),
            "autocorr": self._autocorr,
        }

    def latest_jpeg(self):
        return self._latest_jpeg


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
def create_app(engine: "Engine"):
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)

    @app.route("/")
    def index():
        return Response(_HTML, mimetype="text/html")

    @app.route("/stream.mjpg")
    def stream():
        def gen():
            boundary = b"--frame\r\n"
            while True:
                jpg = engine.latest_jpeg()
                yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(1 / 30)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/snapshot.png")
    def snapshot():
        from PIL import Image
        with engine.lock:
            frame = engine._latest_jpeg
        # re-decode the latest JPEG to a PNG for lossless download
        img = Image.open(io.BytesIO(frame))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return Response(buf.getvalue(), mimetype="image/png",
                        headers={"Content-Disposition": "attachment; filename=tempest.png"})

    @app.route("/api/status")
    def api_status():
        return jsonify(engine.status())

    @app.route("/api/modes")
    def api_modes():
        return jsonify([
            {"name": m.name, "width": m.width, "height": m.height,
             "refresh_rate": m.refresh_rate}
            for m in videomodes.get_video_modes()
        ])

    @app.route("/api/config", methods=["POST"])
    def api_config():
        engine.reconfigure(**(request.get_json(force=True) or {}))
        return jsonify(engine.status())

    @app.route("/api/nudge", methods=["POST"])
    def api_nudge():
        data = request.get_json(force=True) or {}
        engine.nudge(data.get("direction", "reset"), int(data.get("pixels", 1)))
        return jsonify(engine.status())

    @app.route("/api/detect", methods=["POST"])
    def api_detect():
        applied = engine.apply_detection()
        return jsonify({"applied": applied, "status": engine.status()})

    @app.route("/api/source", methods=["POST"])
    def api_source():
        data = request.get_json(force=True) or {}
        kind = data.get("kind")
        try:
            if kind == "synthetic":
                engine.start_synthetic()
            elif kind == "file":
                engine.start_file(data["path"], float(data["samplerate"]),
                                  data.get("format", "uint8"))
            elif kind == "sdr":
                engine.start_sdr(data.get("driver", "rtlsdr"), float(data["samplerate"]),
                                 float(data["frequency"]), data.get("gain", "auto"))
            else:
                return jsonify({"error": f"unknown source kind {kind!r}"}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(engine.status())

    return app


def run(host="127.0.0.1", port=8000, debug=False):
    """Start the web control panel (needs Flask and Pillow)."""
    try:
        import flask  # noqa: F401
        import PIL  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "the web GUI needs Flask and Pillow: pip install flask pillow"
        ) from exc
    engine = Engine()
    app = create_app(engine)
    print(f"TempestSDR web control panel: http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TempestSDR — control panel</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#2d3644;--txt:#c9d4e3;
--dim:#7d8ba0;--acc:#3fb950;--acc2:#58a6ff;--warn:#d29922;--bad:#f85149;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:13px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:10px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:14px}
header h1{font-size:15px;margin:0;font-weight:600;letter-spacing:.3px}
header .tag{color:var(--dim);font-size:12px}
.wrap{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:14px;padding:14px;align-items:start}
@media(max-width:900px){.wrap{grid-template-columns:1fr}}
.view{background:#000;border:1px solid var(--line);border-radius:8px;overflow:hidden;position:relative}
.view img{width:100%;display:block;image-rendering:pixelated;background:#000;min-height:200px}
.badges{position:absolute;top:8px;left:8px;display:flex;gap:6px;flex-wrap:wrap}
.badge{background:rgba(13,17,23,.82);border:1px solid var(--line);border-radius:5px;
padding:3px 8px;font-size:11px;color:var(--dim)}
.badge b{color:var(--txt);font-weight:600}
.badge.lock.on{color:var(--acc);border-color:var(--acc)}
.badge.lock.off{color:var(--warn);border-color:var(--warn)}
.side{display:flex;flex-direction:column;gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px}
.card h2{font-size:12px;text-transform:uppercase;letter-spacing:.6px;color:var(--dim);
margin:0;padding:9px 12px;border-bottom:1px solid var(--line)}
.card .body{padding:11px 12px;display:flex;flex-direction:column;gap:9px}
label{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:12px}
label span.k{color:var(--dim)}
input[type=number],input[type=text],select{background:var(--panel2);border:1px solid var(--line);
color:var(--txt);border-radius:5px;padding:5px 7px;font:inherit;width:130px}
input[type=range]{width:150px;accent-color:var(--acc)}
select{width:100%}
.row{display:flex;gap:7px;flex-wrap:wrap}
button{background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:6px;
padding:6px 10px;font:inherit;cursor:pointer;transition:.12s}
button:hover{border-color:var(--acc2);color:#fff}
button.primary{background:var(--acc);border-color:var(--acc);color:#06210e;font-weight:600}
button.primary:hover{filter:brightness(1.08);color:#06210e}
.toggle{display:flex;align-items:center;gap:7px;cursor:pointer;font-size:12px}
.toggle input{accent-color:var(--acc);width:15px;height:15px}
.pad{display:grid;grid-template-columns:repeat(3,1fr);gap:5px;width:150px;margin-left:auto}
.pad button{padding:6px 0}
canvas{width:100%;height:80px;display:block;background:var(--panel2);border-radius:5px}
.hint{color:var(--dim);font-size:11px}
.acc-info{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);margin-top:5px}
.acc-info b{color:var(--acc2)}
.err{color:var(--bad);font-size:11px;min-height:14px}
.seg{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}
.seg button{border:0;border-radius:0;flex:1;padding:6px 4px}
.seg button.act{background:var(--acc2);color:#06210e;font-weight:600}
</style></head>
<body>
<header>
  <h1>TempestSDR</h1>
  <span class="tag">web control panel · Van Eck reconstruction</span>
</header>
<div class="wrap">
  <div>
    <div class="view">
      <img id="live" src="/stream.mjpg" alt="live reconstruction">
      <div class="badges">
        <div class="badge"><b id="b-geo">–</b></div>
        <div class="badge">SNR <b id="b-snr">–</b></div>
        <div class="badge">FPS <b id="b-fps">–</b></div>
        <div class="badge lock" id="b-lock">unlocked</div>
        <div class="badge" id="b-src">–</div>
      </div>
    </div>
    <div class="card" style="margin-top:12px">
      <h2>Frame-rate autocorrelation — click a peak to set the refresh rate</h2>
      <div class="body">
        <canvas id="acc" width="900" height="80"></canvas>
        <div class="acc-info">
          <span>detected peak: <b id="acc-peak">–</b></span>
          <span class="hint">taller = stronger periodicity</span>
        </div>
      </div>
    </div>
  </div>

  <div class="side">
    <div class="card">
      <h2>Source</h2>
      <div class="body">
        <div class="seg" id="src-seg">
          <button data-k="synthetic" class="act">Synthetic</button>
          <button data-k="file">IQ file</button>
          <button data-k="sdr">Live SDR</button>
        </div>
        <div id="src-synthetic" class="hint">Built-in simulated emanation — no hardware needed.</div>
        <div id="src-file" style="display:none;flex-direction:column;gap:7px">
          <label><span class="k">path</span><input id="f-path" type="text" style="width:180px" placeholder="/path/capture.iq"></label>
          <label><span class="k">sample rate</span><input id="f-sr" type="number" value="8000000"></label>
          <label><span class="k">format</span>
            <select id="f-fmt" style="width:130px">
              <option>uint8</option><option>int8</option><option>int16</option>
              <option>uint16</option><option>float</option></select></label>
        </div>
        <div id="src-sdr" style="display:none;flex-direction:column;gap:7px">
          <label><span class="k">driver</span>
            <select id="s-drv" style="width:150px">
              <option value="rtlsdr">SoapySDR: rtlsdr</option>
              <option value="rtlsdr-native">pyrtlsdr (native)</option>
              <option value="hackrf">SoapySDR: hackrf</option>
              <option value="uhd">SoapySDR: uhd</option></select></label>
          <label><span class="k">sample rate</span><input id="s-sr" type="number" value="8000000"></label>
          <label><span class="k">frequency</span><input id="s-fq" type="number" value="400000000"></label>
          <label><span class="k">gain</span><input id="s-gn" type="text" value="auto"></label>
        </div>
        <button class="primary" id="src-apply">Start source</button>
        <div class="err" id="src-err"></div>
      </div>
    </div>

    <div class="card">
      <h2>Video mode</h2>
      <div class="body">
        <select id="mode-sel"><option value="">— preset —</option></select>
        <label><span class="k">total height (lines)</span><input id="m-h" type="number"></label>
        <label><span class="k">refresh rate (Hz)</span><input id="m-r" type="number" step="0.001"></label>
        <div class="row">
          <button id="mode-apply">Apply</button>
          <button class="primary" id="detect" title="estimate refresh & resolution from the signal">Auto-detect (blind)</button>
        </div>
        <div class="hint" id="detect-out"></div>
      </div>
    </div>

    <div class="card">
      <h2>Processing</h2>
      <div class="body">
        <label><span class="k">motion blur</span><input id="p-mb" type="range" min="0" max="0.95" step="0.05"></label>
        <label class="toggle"><input type="checkbox" id="p-as"> auto-shift (align frame)</label>
        <label class="toggle"><input type="checkbox" id="p-pll"> frame-rate PLL</label>
        <label class="toggle"><input type="checkbox" id="p-nn"> nearest-neighbour resample</label>
        <label class="toggle"><input type="checkbox" id="p-lp"> low-pass before sync</label>
        <label class="toggle"><input type="checkbox" id="p-ag"> auto-gain after processing</label>
      </div>
    </div>

    <div class="card">
      <h2>Manual sync</h2>
      <div class="body">
        <div class="row" style="align-items:center">
          <span class="hint">nudge the image</span>
          <div class="pad">
            <span></span><button data-d="up">▲</button><span></span>
            <button data-d="left">◀</button><button data-d="reset">⟲</button><button data-d="right">▶</button>
            <span></span><button data-d="down">▼</button><span></span>
          </div>
        </div>
        <a href="/snapshot.png" download><button style="width:100%">⤓ Save PNG snapshot</button></a>
      </div>
    </div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
let SR=1, modes=[];

async function jpost(url,body){const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});return r.json();}

// source segmented control
document.querySelectorAll('#src-seg button').forEach(b=>b.onclick=()=>{
  document.querySelectorAll('#src-seg button').forEach(x=>x.classList.remove('act'));
  b.classList.add('act');
  for(const k of ['synthetic','file','sdr']) $('src-'+k).style.display = (k===b.dataset.k)?'flex':'none';
  if(b.dataset.k==='synthetic') $('src-synthetic').style.display='block';
});
$('src-apply').onclick=async()=>{
  const k=document.querySelector('#src-seg button.act').dataset.k; let body={kind:k};
  if(k==='file') body={kind:k,path:$('f-path').value,samplerate:+$('f-sr').value,format:$('f-fmt').value};
  if(k==='sdr') body={kind:k,driver:$('s-drv').value,samplerate:+$('s-sr').value,frequency:+$('s-fq').value,gain:$('s-gn').value};
  const r=await jpost('/api/source',body);
  $('src-err').textContent = r.error? ('⚠ '+r.error):'';
};

// video mode
fetch('/api/modes').then(r=>r.json()).then(ms=>{modes=ms;
  const sel=$('mode-sel'); ms.forEach((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent=m.name;sel.appendChild(o);});
});
$('mode-sel').onchange=e=>{const m=modes[+e.target.value]; if(m){$('m-h').value=m.height;$('m-r').value=m.refresh_rate;}};
$('mode-apply').onclick=()=>jpost('/api/config',{height:+$('m-h').value,refresh_rate:+$('m-r').value});
$('detect').onclick=async()=>{const r=await jpost('/api/detect',{});
  if(r.applied){$('detect-out').textContent=`→ ${r.applied.refresh_rate.toFixed(2)} Hz, ${r.applied.height} lines`;
    $('m-h').value=r.applied.height;$('m-r').value=r.applied.refresh_rate.toFixed(3);}
  else $('detect-out').textContent='not enough signal yet';};

// processing controls
$('p-mb').oninput=()=>jpost('/api/config',{motion_blur:+$('p-mb').value});
const tog=(id,field)=>$(id).onchange=()=>jpost('/api/config',{[field]:$(id).checked});
tog('p-as','autoshift');tog('p-pll','framerate_pll');tog('p-nn','nearest');
tog('p-lp','lowpass_before_sync');tog('p-ag','autogain_after');

// manual sync
document.querySelectorAll('.pad button').forEach(b=>b.onclick=()=>jpost('/api/nudge',{direction:b.dataset.d,pixels:4}));

// autocorrelation plot
const cv=$('acc'),cx=cv.getContext('2d');let acc=null;
function drawAcc(){const w=cv.width,h=cv.height;cx.clearRect(0,0,w,h);
  cx.fillStyle='#1c2330';cx.fillRect(0,0,w,h);
  if(!acc||!acc.values||!acc.values.length){cx.fillStyle='#7d8ba0';cx.font='12px system-ui';cx.fillText('acquiring…',10,45);return;}
  const v=acc.values,n=v.length;cx.strokeStyle='#3fb950';cx.lineWidth=1.5;cx.beginPath();
  for(let i=0;i<n;i++){const x=i/(n-1)*w,y=h-4-v[i]*(h-10);i?cx.lineTo(x,y):cx.moveTo(x,y);}cx.stroke();
  // peak marker
  const pr=(acc.peak_lag-acc.lag_lo)/(acc.lag_hi-acc.lag_lo);const px=pr*w;
  cx.strokeStyle='#58a6ff';cx.setLineDash([3,3]);cx.beginPath();cx.moveTo(px,0);cx.lineTo(px,h);cx.stroke();cx.setLineDash([]);
}
cv.onclick=e=>{if(!acc)return;const r=cv.getBoundingClientRect();const frac=(e.clientX-r.left)/r.width;
  const lag=acc.lag_lo+frac*(acc.lag_hi-acc.lag_lo);const refresh=SR/lag;
  $('m-r').value=refresh.toFixed(3);jpost('/api/config',{refresh_rate:refresh});};

// status poll
let userTouched={};
['p-mb','p-as','p-pll','p-nn','p-lp','p-ag'].forEach(id=>$(id).addEventListener('input',()=>userTouched[id]=Date.now()));
function fresh(id){return (Date.now()-(userTouched[id]||0))>1500;}
async function poll(){try{const s=await fetch('/api/status').then(r=>r.json());
  SR=s.geometry.samplerate;
  $('b-geo').textContent=`${s.geometry.width}×${s.geometry.height} @ ${s.geometry.refresh_rate}Hz`;
  $('b-snr').textContent=s.snr;$('b-fps').textContent=s.fps;
  const bl=$('b-lock');bl.textContent=s.locked?'LOCKED':'unlocked';bl.className='badge lock '+(s.locked?'on':'off');
  $('b-src').textContent=s.source_kind;
  if(fresh('p-mb'))$('p-mb').value=s.params.motion_blur;
  if(fresh('p-as'))$('p-as').checked=s.params.autoshift;
  if(fresh('p-pll'))$('p-pll').checked=s.params.framerate_pll;
  if(fresh('p-nn'))$('p-nn').checked=s.params.nearest;
  if(fresh('p-lp'))$('p-lp').checked=s.params.lowpass_before_sync;
  if(fresh('p-ag'))$('p-ag').checked=s.params.autogain_after;
  if(document.activeElement!==$('m-h')&&!$('m-h').value)$('m-h').value=s.geometry.height;
  if(document.activeElement!==$('m-r')&&!$('m-r').value)$('m-r').value=s.geometry.refresh_rate;
  acc=s.autocorr;drawAcc();
  $('acc-peak').textContent=acc?`${acc.peak_refresh.toFixed(2)} Hz (lag ${acc.peak_lag})`:'–';
}catch(e){}setTimeout(poll,500);}
poll();
</script>
</body></html>"""

"""Remote control server for a fleet of TempestSDR probes.

Edge probes (see :mod:`tempestsdr.agent`) reconstruct frames locally and POST
them here with a little metadata; this server keeps the latest frame per device,
shows a multi-probe dashboard, and hands queued commands back to each probe when
it next checks in.

Transport is plain HTTP + JSON so probes can run the dependency-light
:mod:`tempestsdr.agent` (standard-library ``urllib`` only).  A shared
``X-API-Key`` header authenticates probes when an API key is configured.

Intended for authorised measurement only — deploy probes against equipment you
own or are explicitly permitted to test.

Launch with ``tempestsdr server`` and open http://127.0.0.1:9000.
"""

from __future__ import annotations

import base64
import json
import os
import threading
import time
from collections import deque

# ``time.time`` is used for wall-clock "last seen" bookkeeping.


class Fleet:
    """In-memory registry of probes and their pending command queues."""

    def __init__(self, api_key: str | None = None, archive_dir: str | None = None):
        self.api_key = api_key
        self.archive_dir = archive_dir
        self._lock = threading.Lock()
        self._devices: dict[str, dict] = {}
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)

    def _dev(self, device_id: str) -> dict:
        d = self._devices.get(device_id)
        if d is None:
            d = {"id": device_id, "first_seen": time.time(), "last_seen": 0.0,
                 "meta": {}, "frame": None, "frame_ts": 0.0, "commands": deque()}
            self._devices[device_id] = d
        return d

    def ingest(self, device_id: str, meta: dict, frame: bytes | None) -> list[dict]:
        """Store a frame + metadata; return any queued commands for the probe."""
        with self._lock:
            d = self._dev(device_id)
            d["last_seen"] = time.time()
            d["meta"] = meta or {}
            if frame:
                d["frame"] = frame
                d["frame_ts"] = d["last_seen"]
                if self.archive_dir:
                    self._archive(device_id, frame)
            cmds = list(d["commands"])
            d["commands"].clear()
        return cmds

    def _archive(self, device_id: str, frame: bytes) -> None:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in device_id)
        day = time.strftime("%Y%m%d", time.gmtime())
        folder = os.path.join(self.archive_dir, safe, day)
        os.makedirs(folder, exist_ok=True)
        stamp = time.strftime("%H%M%S", time.gmtime())
        with open(os.path.join(folder, f"{stamp}.jpg"), "wb") as fh:
            fh.write(frame)

    def queue_command(self, device_id: str, command: dict) -> None:
        with self._lock:
            self._dev(device_id)["commands"].append(command)

    def devices(self) -> list[dict]:
        now = time.time()
        with self._lock:
            out = []
            for d in self._devices.values():
                out.append({
                    "id": d["id"], "meta": d["meta"],
                    "last_seen": d["last_seen"],
                    "age": round(now - d["last_seen"], 1) if d["last_seen"] else None,
                    "online": bool(d["last_seen"] and now - d["last_seen"] < 15),
                    "has_frame": d["frame"] is not None,
                    "pending": len(d["commands"]),
                })
            return sorted(out, key=lambda x: x["id"])

    def frame(self, device_id: str) -> bytes | None:
        with self._lock:
            d = self._devices.get(device_id)
            return d["frame"] if d else None


def create_app(fleet: "Fleet"):
    from flask import Flask, Response, jsonify, request, abort

    app = Flask(__name__)

    def _check_auth():
        if fleet.api_key and request.headers.get("X-API-Key") != fleet.api_key:
            abort(401)

    @app.route("/api/ingest", methods=["POST"])
    def ingest():
        _check_auth()
        data = request.get_json(force=True, silent=True) or {}
        device_id = data.get("device_id")
        if not device_id:
            return jsonify({"error": "device_id required"}), 400
        frame = None
        if data.get("frame_b64"):
            try:
                frame = base64.b64decode(data["frame_b64"])
            except Exception:
                return jsonify({"error": "bad frame_b64"}), 400
        cmds = fleet.ingest(device_id, data.get("meta", {}), frame)
        return jsonify({"ok": True, "commands": cmds})

    @app.route("/api/devices")
    def devices():
        return jsonify(fleet.devices())

    @app.route("/api/frame/<device_id>")
    def frame(device_id):
        f = fleet.frame(device_id)
        if not f:
            abort(404)
        return Response(f, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/api/command/<device_id>", methods=["POST"])
    def command(device_id):
        _check_auth()
        cmd = request.get_json(force=True, silent=True) or {}
        if "type" not in cmd:
            return jsonify({"error": "command 'type' required"}), 400
        fleet.queue_command(device_id, cmd)
        return jsonify({"ok": True})

    @app.route("/")
    def index():
        return Response(_HTML, mimetype="text/html")

    return app


def run(host="0.0.0.0", port=9000, api_key=None, archive_dir=None, debug=False):
    """Start the control server (needs Flask)."""
    try:
        import flask  # noqa: F401
    except ImportError as exc:
        raise ImportError("the control server needs Flask: pip install flask") from exc
    fleet = Fleet(api_key=api_key, archive_dir=archive_dir)
    app = create_app(fleet)
    print(f"TempestSDR control server: http://{host}:{port}"
          + (f"  (archiving to {archive_dir})" if archive_dir else ""))
    app.run(host=host, port=port, debug=debug, threaded=True, use_reloader=False)


_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TempestSDR — control server</title>
<style>
:root{--bg:#0d1117;--panel:#161b22;--panel2:#1c2330;--line:#2d3644;--txt:#c9d4e3;
--dim:#7d8ba0;--acc:#3fb950;--acc2:#58a6ff;--bad:#f85149;--warn:#d29922;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--txt);font:13px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
header{padding:11px 16px;border-bottom:1px solid var(--line);display:flex;align-items:center;gap:12px}
header h1{font-size:15px;margin:0;font-weight:600}
header .tag{color:var(--dim);font-size:12px}
header .count{margin-left:auto;color:var(--dim)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:14px;padding:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;overflow:hidden}
.card .top{display:flex;align-items:center;gap:8px;padding:8px 11px;border-bottom:1px solid var(--line)}
.dot{width:9px;height:9px;border-radius:50%;background:var(--bad)}
.dot.on{background:var(--acc);box-shadow:0 0 6px var(--acc)}
.card .id{font-weight:600}
.card .age{margin-left:auto;color:var(--dim);font-size:11px}
.card img{width:100%;display:block;background:#000;image-rendering:pixelated;aspect-ratio:16/10;object-fit:contain}
.meta{display:flex;flex-wrap:wrap;gap:5px;padding:8px 11px;font-size:11px;color:var(--dim)}
.chip{background:var(--panel2);border:1px solid var(--line);border-radius:5px;padding:2px 7px}
.chip b{color:var(--txt)}
.chip.lock b{color:var(--acc)}
.ctl{display:flex;flex-wrap:wrap;gap:6px;padding:9px 11px;border-top:1px solid var(--line)}
.ctl input,.ctl select{background:var(--panel2);border:1px solid var(--line);color:var(--txt);
border-radius:5px;padding:4px 6px;font:inherit;width:96px}
.ctl button{background:var(--panel2);border:1px solid var(--line);color:var(--txt);border-radius:5px;
padding:4px 9px;font:inherit;cursor:pointer}
.ctl button:hover{border-color:var(--acc2);color:#fff}
.ctl button.p{background:var(--acc2);border-color:var(--acc2);color:#06210e;font-weight:600}
.empty{color:var(--dim);padding:40px;text-align:center;grid-column:1/-1}
.row{display:flex;gap:6px;align-items:center;width:100%}
</style></head>
<body>
<header><h1>TempestSDR</h1><span class="tag">remote control server · probe fleet</span>
<span class="count" id="count">0 probes</span></header>
<div class="grid" id="grid"><div class="empty">waiting for probes to check in…</div></div>
<script>
const $=id=>document.getElementById(id);
async function jpost(u,b){return fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});}
function card(d){
  const m=d.meta||{},g=m.geometry||{};
  const el=document.createElement('div');el.className='card';el.id='c-'+d.id;
  el.innerHTML=`
   <div class="top"><span class="dot ${d.online?'on':''}"></span>
     <span class="id">${d.id}</span><span class="age">${d.age==null?'never':d.age+'s ago'}</span></div>
   <img src="/api/frame/${encodeURIComponent(d.id)}?t=0" alt="no frame" onerror="this.style.opacity=.15">
   <div class="meta">
     <span class="chip">${g.width||'?'}×${g.height||'?'} @ ${g.refresh_rate||'?'}Hz</span>
     <span class="chip">SNR <b>${m.snr??'–'}</b></span>
     <span class="chip lock"><b>${m.locked?'LOCKED':'unlocked'}</b></span>
     <span class="chip">src <b>${m.source_kind||'?'}</b></span>
     ${m.freq?`<span class="chip">${(m.freq/1e6).toFixed(2)} MHz</span>`:''}
   </div>
   <div class="ctl">
     <div class="row"><input type="number" id="f-${d.id}" placeholder="freq Hz" style="flex:1">
       <button data-a="freq" data-id="${d.id}">Set freq</button></div>
     <div class="row"><input type="number" id="h-${d.id}" placeholder="height">
       <input type="number" id="r-${d.id}" placeholder="refresh" step="0.01">
       <button data-a="mode" data-id="${d.id}">Set mode</button></div>
     <div class="row"><button class="p" data-a="detect" data-id="${d.id}">Auto-detect</button>
       <button data-a="synthetic" data-id="${d.id}">Synthetic</button>
       <button data-a="reset" data-id="${d.id}">Reset sync</button></div>
   </div>`;
  return el;
}
async function act(a,id){
  if(a==='freq') await jpost('/api/command/'+id,{type:'set_freq',freq:+$('f-'+id).value});
  else if(a==='mode') await jpost('/api/command/'+id,{type:'set_mode',height:+$('h-'+id).value,refresh_rate:+$('r-'+id).value});
  else if(a==='detect') await jpost('/api/command/'+id,{type:'detect'});
  else if(a==='synthetic') await jpost('/api/command/'+id,{type:'set_source',kind:'synthetic'});
  else if(a==='reset') await jpost('/api/command/'+id,{type:'nudge',direction:'reset'});
}
document.addEventListener('click',e=>{const b=e.target.closest('button[data-a]');if(b)act(b.dataset.a,b.dataset.id);});
let known=new Set();
async function tick(){try{const ds=await fetch('/api/devices').then(r=>r.json());
  $('count').textContent=ds.length+' probe'+(ds.length!==1?'s':'');
  const grid=$('grid');
  if(!ds.length){if(!grid.querySelector('.empty'))grid.innerHTML='<div class="empty">waiting for probes to check in…</div>';}
  else{const empty=grid.querySelector('.empty');if(empty)empty.remove();}
  const ids=new Set(ds.map(d=>d.id));
  for(const id of known)if(!ids.has(id)){const e=$('c-'+id);if(e)e.remove();}
  for(const d of ds){let e=$('c-'+d.id);
    if(!e){e=card(d);grid.appendChild(e);}
    else{ // update dynamic bits
      e.querySelector('.dot').className='dot '+(d.online?'on':'');
      e.querySelector('.age').textContent=d.age==null?'never':d.age+'s ago';
      const m=d.meta||{},g=m.geometry||{};
      const chips=e.querySelectorAll('.meta .chip');
      chips[0].innerHTML=`${g.width||'?'}×${g.height||'?'} @ ${g.refresh_rate||'?'}Hz`;
      chips[1].innerHTML=`SNR <b>${m.snr??'–'}</b>`;
      chips[2].innerHTML=`<b>${m.locked?'LOCKED':'unlocked'}</b>`;
      if(d.has_frame){const img=e.querySelector('img');img.style.opacity=1;img.src='/api/frame/'+encodeURIComponent(d.id)+'?t='+Date.now();}
    }
  }
  known=ids;
}catch(e){}setTimeout(tick,1500);}
tick();
</script>
</body></html>"""

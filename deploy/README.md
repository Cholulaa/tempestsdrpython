# Deploying TempestSDR as an autonomous probe fleet

This directory has everything needed to run TempestSDR unattended: edge
**probes** (e.g. Raspberry Pi + RTL-SDR) that reconstruct locally and stream the
recovered frames to a central **control server** with a fleet dashboard.

```
   ┌──────────────┐   reconstruct     ┌──────────────┐  HTTP + JSON   ┌────────────────┐
   │  target      │  ~~emanations~~▶  │  probe (Pi)  │ ──frames/meta─▶ │ control server │
   │  display     │                   │  RTL-SDR +   │ ◀──commands──── │  + dashboard   │
   └──────────────┘                   │  tempestsdr  │                 └────────────────┘
                                       └──────────────┘
```

> **Authorised use only.** Deploy probes against displays you own or are
> explicitly permitted to test (TEMPEST assessments, sanctioned red-team work,
> measuring your own facility's shielding). Intercepting other people's
> emanations is illegal in most jurisdictions.

## Try it with no hardware first

On any machine:

```bash
pip install -e . flask pillow

# terminal 1 — the control server (dashboard at http://127.0.0.1:9000)
tempestsdr server --port 9000

# terminal 2 — a probe using the built-in synthetic source
tempestsdr agent --server http://127.0.0.1:9000 --device-id test-01 --synthetic
```

Open http://127.0.0.1:9000 — the probe appears with its live reconstruction and
you can retune / change mode / auto-detect it from the dashboard.

## Control server (any always-on host)

```bash
pip install tempestsdr flask
tempestsdr server --host 0.0.0.0 --port 9000 \
    --api-key "$(openssl rand -hex 16)" \
    --archive /var/lib/tempestsdr/frames     # optional: keep every frame
```

Install as a service with `tempestsdr-server.service` (set `TEMPEST_API_KEY`).

## Probe (Raspberry Pi + RTL-SDR)

```bash
sudo apt install rtl-sdr python3-pip
pip install tempestsdr pyrtlsdr pillow      # numpy/scipy pulled in automatically

sudo mkdir -p /opt/tempestsdr /etc/tempestsdr
sudo cp -r . /opt/tempestsdr/
sudo cp agent.env /etc/tempestsdr/agent.env
sudo chmod +x /opt/tempestsdr/deploy/run-agent.sh
sudoedit /etc/tempestsdr/agent.env          # set SERVER, DEVICE_ID, API_KEY, FREQUENCY

sudo cp tempestsdr-agent.service /etc/systemd/system/
sudo systemctl enable --now tempestsdr-agent
journalctl -u tempestsdr-agent -f           # watch it check in
```

The probe keeps reconstructing and buffering through network outages and
reconnects with backoff; `systemd` restarts it if it ever dies.

### Raspberry Pi models

The probe is deliberately light. On the edge it needs only `numpy`, `pillow` and
a radio driver (`pyrtlsdr` for RTL-SDR); `scipy` is not required at all, and
Flask is only used by the control server, never by the probe.

Raspberry Pi 4 or Pi 3 (quad-core, 1 GB or more of RAM) reconstruct in real time
at RTL-SDR rates:

```bash
sudo apt install python3-numpy python3-pil
pip install "pyrtlsdr<0.3"
pip install .
```

Raspberry Pi 1 or Pi Zero (single-core ARMv6, little RAM) work too, at a lower
frame rate. A few tips:

- Install `numpy` from piwheels (the default index on Raspberry Pi OS) so it
  resolves a prebuilt wheel instead of compiling from source.
- Use a lower sample rate (for example `SAMPLERATE=1200000` or `900000` in
  `agent.env`). Horizontal resolution drops but the board keeps up.
- Raise the report interval (`INTERVAL=5`) and lower `QUALITY=60` to cut the
  upload and JPEG work.
- If the board still cannot keep pace, the SDR just drops samples and the frame
  rate falls; memory never grows unbounded.

The whole toolkit targets Python 3.7 and later, which covers the Python shipped
on older Raspberry Pi OS releases.

### RTL-SDR bandwidth reality

An RTL-SDR reliably streams **~2.4 Msps**. The reconstructable horizontal pixel
clock scales with the sample rate, so RTL-SDR probes suit **lower-resolution /
lower-refresh** targets and produce coarse (but often readable) images. For
higher resolutions use a wider front-end (HackRF, USRP, Airspy) via the
SoapySDR driver — set `DRIVER=` accordingly in `agent.env`.

### Finding the frequency

The hardest part is finding where the target leaks. Tune near a harmonic of the
pixel clock and look for a strong, plausible frame rate. `tempestsdr detect`
reports a **confidence** (frame-rate autocorrelation peak prominence): `~1` means
noise, well above `~2` means a real periodic video signal.

`scripts/scan.ps1` (Windows) automates the sweep — it records a short capture at
each frequency and ranks them by confidence:

```powershell
.\scripts\scan.ps1 -Start 250e6 -Stop 600e6 -Step 25e6
```

Then reconstruct at the best candidate frequency with the suggested preset.

## Runtime control

From the dashboard (or `POST /api/command/<device_id>`) you can push, per probe:

| command | effect |
| --- | --- |
| `set_freq` | retune the SDR |
| `set_mode` | set total height + refresh rate |
| `detect` | blind auto-detect and apply the video mode |
| `set_config` | motion blur, autoshift, PLL, resampling, ... |
| `set_source` | switch between synthetic / SDR / file |
| `nudge` | manual sync (or `reset`) |

## Security notes

- Put the control server behind TLS (a reverse proxy) and require `--api-key`;
  the dev server shipped here is plaintext HTTP and single-process.
- Frames contain reconstructed screen content — treat the server and its
  `--archive` directory as sensitive, and restrict network access to it.

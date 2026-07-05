"""Command-line interface for the TempestSDR Python port.

Sub-commands::

    tempestsdr list-modes                 # print the VESA presets
    tempestsdr synth  IMAGE OUT.iq ...     # forward-model a capture from an image
    tempestsdr detect CAPTURE.iq ...       # estimate refresh rate / line count
    tempestsdr reconstruct CAPTURE.iq ...  # capture -> PNG frame(s)
    tempestsdr demo   IMAGE OUT.png ...    # image -> synth -> reconstruct (no HW)
    tempestsdr live   ...                  # real-time from RTL-SDR / SoapySDR

Run ``tempestsdr <command> -h`` for the options of each command.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from . import videomodes
from .processor import ProcessorConfig, TempestProcessor


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _save_png(frame: np.ndarray, path: str) -> None:
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("Pillow is required to write PNGs (pip install pillow)")
    arr = np.clip(frame, 0.0, 1.0)
    img = (arr * 255.0).round().astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


def _resolve_geometry(args) -> tuple[int, float]:
    """Return (total_height, refresh_rate) from --mode or --height/--refresh."""
    if getattr(args, "mode", None):
        mode = videomodes.find_by_name(args.mode)
        if mode is None:
            raise SystemExit(f"unknown video mode {args.mode!r}; try 'list-modes'")
        return mode.height, mode.refresh_rate
    if args.height is None or args.refresh is None:
        raise SystemExit("specify either --mode NAME or both --height and --refresh")
    return args.height, args.refresh


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_list_modes(args) -> int:
    print(f"{'name':<40} {'total_w':>8} {'total_h':>8} {'refresh':>8}")
    print("-" * 68)
    for mode in videomodes.get_video_modes():
        print(f"{mode.name:<40} {mode.width:>8} {mode.height:>8} {mode.refresh_rate:>8.0f}")
    return 0


def cmd_synth(args) -> int:
    from .synth import SyntheticConfig, generate, load_image
    from .sources.file_source import save_iq

    height, refresh = _resolve_geometry(args)
    mode = videomodes.find_by_name(args.mode) if args.mode else None
    width = mode.width if mode else args.width
    if width is None:
        raise SystemExit("specify --mode or --width for the synthetic total line width")

    image = load_image(args.image)
    cfg = SyntheticConfig(
        total_width=width, total_height=height, refresh_rate=refresh,
        samplerate=args.samplerate, num_frames=args.frames, snr_db=args.snr,
        freq_offset=args.freq_offset, start_offset_pixels=args.start_offset,
        mode=args.emod, seed=args.seed,
    )
    iq = generate(image, cfg)
    save_iq(args.output, iq, sample_format=args.format)
    print(f"wrote {iq.size} samples ({args.format}) to {args.output}")
    print(f"  geometry: total {width}x{height} @ {refresh:g}Hz, "
          f"samplerate {args.samplerate:g}")
    return 0


def cmd_detect(args) -> int:
    from .sources.file_source import FileSource
    from .framerate import FrameRateDetector

    src = FileSource(args.capture, samplerate=args.samplerate,
                     sample_format=args.format, max_samples=args.max_samples)
    from . import dsp
    detector = FrameRateDetector(args.samplerate)
    for block in src:
        detector.run(dsp.am_demodulate(block))
    est = detector.estimate_resolution()
    if est is None:
        print("could not estimate a frame rate (not enough data?)")
        return 1
    print(f"refresh rate     : {est['refresh_rate']:.3f} Hz")
    print(f"frame period     : {est['frame_period_samples']:.1f} samples")
    print(f"line period      : {est['line_period_samples']:.2f} samples")
    print(f"estimated lines  : {est['height_lines']:.1f}")
    closest = videomodes.find_closest(est["refresh_rate"], round(est["height_lines"]))
    if closest:
        print(f"closest preset   : {closest.name}")
    return 0


def _build_processor(args, samplerate) -> TempestProcessor:
    height, refresh = _resolve_geometry(args)
    cfg = ProcessorConfig(
        samplerate=samplerate, height=height, refresh_rate=refresh,
        autoshift=not args.no_autoshift, nearest=args.nearest,
        motion_blur=args.motion_blur, lowpass_before_sync=args.lowpass_before_sync,
        autogain_after=args.autogain_after,
    )
    return TempestProcessor(cfg)


def cmd_reconstruct(args) -> int:
    from .sources.file_source import FileSource

    src = FileSource(args.capture, samplerate=args.samplerate,
                     sample_format=args.format, max_samples=args.max_samples)
    proc = _build_processor(args, args.samplerate)
    print(f"frame geometry: {proc.width}x{proc.config.height} "
          f"({proc.pixels_per_frame} px/frame)")

    frames: list[np.ndarray] = []
    for block in src:
        frames.extend(proc.process(block))
    if not frames:
        print("no complete frames were produced (capture too short?)")
        return 1

    if args.all_frames:
        base = args.output.rsplit(".", 1)[0]
        for i, frame in enumerate(frames):
            _save_png(frame, f"{base}_{i:03d}.png")
        print(f"wrote {len(frames)} frames to {base}_NNN.png")
    else:
        _save_png(frames[-1], args.output)
        print(f"wrote final frame ({len(frames)} produced) to {args.output}")
    return 0


def cmd_demo(args) -> int:
    from .synth import SyntheticConfig, generate, load_image

    height, refresh = _resolve_geometry(args)
    mode = videomodes.find_by_name(args.mode) if args.mode else None
    width = mode.width if mode else args.width
    if width is None:
        raise SystemExit("specify --mode or --width")

    image = load_image(args.image)
    scfg = SyntheticConfig(
        total_width=width, total_height=height, refresh_rate=refresh,
        samplerate=args.samplerate, num_frames=args.frames, snr_db=args.snr,
        start_offset_pixels=args.start_offset, mode=args.emod, seed=args.seed,
    )
    iq = generate(image, scfg)

    proc = _build_processor(args, args.samplerate)
    frames = proc.process(iq)
    if not frames:
        print("reconstruction produced no frames")
        return 1
    _save_png(frames[-1], args.output)
    print(f"synthesised {iq.size} samples, reconstructed {len(frames)} frames")
    print(f"wrote reconstruction to {args.output}")
    return 0


def cmd_live(args) -> int:  # pragma: no cover - requires hardware
    if args.driver == "rtlsdr-native":
        from .sources.rtlsdr_source import RtlSdrSource
        src = RtlSdrSource(args.samplerate, args.frequency, gain=args.gain)
    else:
        from .sources.soapy_source import SoapySource
        src = SoapySource(args.samplerate, args.frequency, driver=args.driver,
                          gain=None if args.gain == "auto" else float(args.gain))
    proc = _build_processor(args, args.samplerate)
    from .gui import run_live_gui
    run_live_gui(proc, src)
    return 0


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------
def _add_geometry(p) -> None:
    p.add_argument("--mode", help="video mode preset name (see 'list-modes')")
    p.add_argument("--height", type=int, help="total lines per frame (incl. blanking)")
    p.add_argument("--refresh", type=float, help="vertical refresh rate (Hz)")


def _add_common_proc(p) -> None:
    p.add_argument("--no-autoshift", action="store_true", help="do not auto-align frames")
    p.add_argument("--nearest", action="store_true", help="nearest-neighbour resampling")
    p.add_argument("--motion-blur", type=float, default=0.0,
                   help="frame-averaging coefficient in [0,1) (default 0)")
    p.add_argument("--lowpass-before-sync", action="store_true")
    p.add_argument("--autogain-after", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tempestsdr", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list-modes", help="list VESA video-mode presets")
    p.set_defaults(func=cmd_list_modes)

    p = sub.add_parser("synth", help="forward-model a capture from an image")
    p.add_argument("image")
    p.add_argument("output")
    _add_geometry(p)
    p.add_argument("--width", type=int, help="total line width if not using --mode")
    p.add_argument("--samplerate", type=float, required=True)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--freq-offset", type=float, default=0.0)
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--emod", choices=["amplitude", "edge"], default="amplitude")
    p.add_argument("--format", default="uint8",
                   choices=["float", "int8", "uint8", "int16", "uint16"])
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("detect", help="estimate refresh rate and line count")
    p.add_argument("capture")
    p.add_argument("--samplerate", type=float, required=True)
    p.add_argument("--format", default="uint8",
                   choices=["float", "int8", "uint8", "int16", "uint16"])
    p.add_argument("--max-samples", type=int, default=4_000_000)
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("reconstruct", help="reconstruct image(s) from a capture")
    p.add_argument("capture")
    p.add_argument("output")
    _add_geometry(p)
    _add_common_proc(p)
    p.add_argument("--samplerate", type=float, required=True)
    p.add_argument("--format", default="uint8",
                   choices=["float", "int8", "uint8", "int16", "uint16"])
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--all-frames", action="store_true", help="write every frame")
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("demo", help="image -> synthetic capture -> reconstruction")
    p.add_argument("image")
    p.add_argument("output")
    _add_geometry(p)
    _add_common_proc(p)
    p.add_argument("--width", type=int)
    p.add_argument("--samplerate", type=float, required=True)
    p.add_argument("--frames", type=int, default=6)
    p.add_argument("--snr", type=float, default=10.0)
    p.add_argument("--start-offset", type=int, default=0)
    p.add_argument("--emod", choices=["amplitude", "edge"], default="amplitude")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("live", help="real-time reconstruction from an SDR")
    _add_geometry(p)
    _add_common_proc(p)
    p.add_argument("--driver", default="rtlsdr",
                   help="SoapySDR driver, or 'rtlsdr-native' for pyrtlsdr")
    p.add_argument("--samplerate", type=float, required=True)
    p.add_argument("--frequency", type=float, required=True)
    p.add_argument("--gain", default="auto")
    p.set_defaults(func=cmd_live)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

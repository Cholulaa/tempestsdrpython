#!/usr/bin/env python3
"""Self-contained, no-hardware demonstration of the TempestSDR pipeline.

Generates a test image, forward-models a compromising emanation from it,
estimates the video parameters blind, reconstructs the picture, and writes the
before/after PNGs.  Run with::

    python examples/demo.py

Requires numpy, scipy and pillow.
"""

from __future__ import annotations

import numpy as np

from tempestsdr import dsp
from tempestsdr.framerate import FrameRateDetector
from tempestsdr.processor import ProcessorConfig, TempestProcessor
from tempestsdr.synth import SyntheticConfig, generate
from tempestsdr.videomodes import find_by_name, find_closest


def make_image(h=240, w=320):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fallback: geometric pattern if Pillow is unavailable.
        img = np.zeros((h, w))
        img[20:60, 20:w - 20] = 1.0
        img[80:200, 20:40] = 0.8
        img[h - 30:h, :] = np.linspace(0, 1, w)
        return img
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    d.rectangle([15, 15, w - 15, 55], fill=255)
    d.text((25, 25), "TEMPEST SDR - Python", fill=0)
    d.text((25, 80), "Van Eck phreaking demo", fill=210)
    d.text((25, 110), "no radio hardware required", fill=180)
    d.rectangle([25, 150, w - 25, 195], outline=255, width=2)
    for i in range(w - 60):
        im.putpixel((25 + i, 215), int(255 * i / (w - 60)))
    return np.asarray(im, dtype=np.float64)


def main():
    mode = find_by_name("800x600 @ 60Hz")
    samplerate = mode.width * mode.height * mode.refresh_rate / 2  # -> width matches

    image = make_image()
    print(f"target monitor mode : {mode.name} (total {mode.width}x{mode.height})")
    print(f"receiver sample rate: {samplerate/1e6:.3f} Msps")

    # 1. Forward-model the emanation (this stands in for an SDR capture).
    scfg = SyntheticConfig(
        total_width=mode.width, total_height=mode.height, refresh_rate=mode.refresh_rate,
        samplerate=samplerate, num_frames=12, snr_db=12.0, start_offset_pixels=5000, seed=1,
    )
    iq = generate(image, scfg)
    print(f"synthesised capture : {iq.size} IQ samples")

    # 2. Estimate the video parameters *blind* from the capture.
    det = FrameRateDetector(samplerate)
    det.run(dsp.am_demodulate(iq))
    est = det.estimate_resolution()
    if est:
        guess = find_closest(est["refresh_rate"], round(est["height_lines"]))
        print(f"blind estimate      : {est['refresh_rate']:.2f} Hz, "
              f"{est['height_lines']:.0f} lines -> closest preset "
              f"{guess.name if guess else '?'}")

    # 3. Reconstruct the picture.
    proc = TempestProcessor(ProcessorConfig(
        samplerate=samplerate, height=mode.height, refresh_rate=mode.refresh_rate,
        motion_blur=0.6))
    frames = proc.process(iq)
    print(f"reconstructed       : {len(frames)} frames of {proc.width}x{mode.height}")

    try:
        from PIL import Image
        Image.fromarray((image / image.max() * 255).astype(np.uint8), "L").save("demo_original.png")
        out = np.clip(frames[-1], 0, 1)
        Image.fromarray((out * 255).astype(np.uint8), "L").save("demo_reconstructed.png")
        print("wrote demo_original.png and demo_reconstructed.png")
    except ImportError:
        print("(install pillow to write PNG output)")


if __name__ == "__main__":
    main()

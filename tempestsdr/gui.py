"""Optional real-time viewer.

A lightweight matplotlib live display, the Python analogue of the Java
``ImageVisualizer``.  It is intentionally dependency-light: matplotlib is only
imported when the GUI is actually launched, so headless installs are unaffected.

For serious real-time use with fast SDRs a Qt/OpenGL front-end would be more
appropriate; this viewer targets clarity over frame rate.
"""

from __future__ import annotations

import threading

import numpy as np


def run_live_gui(processor, source, refresh_ms: int = 50):  # pragma: no cover - UI
    """Display frames from ``source`` reconstructed by ``processor`` live."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
    except ImportError as exc:
        raise ImportError("matplotlib is required for the live GUI (pip install matplotlib)") from exc

    latest = {"frame": np.zeros(processor.frame_shape, dtype=np.float32)}
    running = {"on": True}

    def worker():
        try:
            for block in source:
                if not running["on"]:
                    break
                for frame in processor.process(block):
                    latest["frame"] = frame
        finally:
            running["on"] = False

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    fig, ax = plt.subplots()
    im = ax.imshow(latest["frame"], cmap="gray", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_title("TempestSDR — live reconstruction")
    ax.axis("off")

    def update(_frame):
        im.set_data(latest["frame"])
        return (im,)

    _anim = FuncAnimation(fig, update, interval=refresh_ms, blit=True, cache_frame_data=False)
    try:
        plt.show()
    finally:
        running["on"] = False
        if hasattr(source, "stop"):
            source.stop()
        if hasattr(source, "close"):
            source.close()


def frames_to_gif(frames, path: str, fps: int = 10):
    """Save a list of [0,1] frames as an animated GIF (needs Pillow)."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required to write GIFs") from exc
    if not frames:
        raise ValueError("no frames to write")
    images = [
        Image.fromarray((np.clip(f, 0, 1) * 255).round().astype(np.uint8), mode="L")
        for f in frames
    ]
    images[0].save(
        path, save_all=True, append_images=images[1:],
        duration=int(1000 / fps), loop=0,
    )

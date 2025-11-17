import time
from pathlib import Path
from panda3d.core import PNMImage, Filename
from pvp.utils.utils import get_time_str


class Screenshotter:
    """
    Throttled Panda3D window capture that names files with get_time_str().
    Call `maybe(engine)` after you render; it saves at most once per `interval`.
    """
    def __init__(self, directory="ima_log", interval=3.0, prefix="shot"):
        self.directory = Path(directory)
        self.interval = float(interval)
        self.prefix = prefix
        self._last_t = 0.0
        self._warned = False
        self.directory.mkdir(parents=True, exist_ok=True)

    def maybe(self, engine) -> None:
        try:
            win = getattr(engine, "win", None)
            if win is None:
                return
            now = time.time()                          # uses utils.py's `import time`
            if now - self._last_t < self.interval:
                return
            self._last_t = now

            # timestamped filename via your existing helper
            fname = self.directory / f"{self.prefix}_{get_time_str()}.png"
            img = PNMImage()
            if win.getScreenshot(img):
                img.write(Filename.from_os_specific(str(fname)))
        except Exception as e:
            if not self._warned:
                print(f"[Screenshot] Failed to save: {e}")
                self._warned = True

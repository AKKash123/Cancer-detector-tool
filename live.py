"""Real-time video processor for streamlit-webrtc (runs the OpenCV pipeline per frame)."""
import threading
import time
import cv2
import numpy as np
import av
from streamlit_webrtc import VideoProcessorBase
from processing import Params, analyze, resize_max

LEVEL_COLOR = {"Low": (0, 200, 0), "Moderate": (0, 200, 255), "High": (0, 0, 255)}


class LiveProcessor(VideoProcessorBase):
    def __init__(self):
        self.params = Params(denoise=False)   # denoise is slow; off by default for live
        self.mirror = True
        self.view = "Annotated"               # Annotated | Enhanced | Red mask | White mask
        self.max_side = 480
        self._ema = None
        self._t = time.time()
        self._fps = 0.0
        self._lock = threading.Lock()
        self.last_frame = None
        self.last_stats = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        if self.mirror:
            img = cv2.flip(img, 1)
        img = resize_max(img, self.max_side)
        res = analyze(img, self.params)

        # Smooth the score over time so the label does not flicker
        sc = res.stats["score"]
        self._ema = sc if self._ema is None else 0.8 * self._ema + 0.2 * sc
        level = "Low" if self._ema < 25 else "Moderate" if self._ema < 55 else "High"

        if self.view == "Enhanced":
            out = res.enhanced.copy()
        elif self.view == "Red mask":
            out = cv2.cvtColor(res.red_mask, cv2.COLOR_GRAY2BGR)
        elif self.view == "White mask":
            out = cv2.cvtColor(res.white_mask, cv2.COLOR_GRAY2BGR)
        else:
            out = res.annotated

        now = time.time()
        self._fps = 0.9 * self._fps + 0.1 * (1 / max(now - self._t, 1e-3))
        self._t = now
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(out, f"Attention: {level} ({self._ema:.0f})  R:{res.stats['red_pct']}%  "
                    f"W:{res.stats['white_pct']}%  {self._fps:.0f} fps",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEVEL_COLOR[level], 1, cv2.LINE_AA)
        with self._lock:
            self.last_frame = out.copy()
            self.last_stats = dict(res.stats, level=level, score=round(self._ema, 1))
        return av.VideoFrame.from_ndarray(out, format="bgr24")
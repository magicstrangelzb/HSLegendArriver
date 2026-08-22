"""Read-only desktop capture for the visible HSAng recommendation panel."""

from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
import time
import uuid
from typing import Callable, Optional

import cv2
import numpy as np
from PIL import ImageGrab

from src.recommendation_config import RecommendationConfig
from src.recommendation_models import FrameEvidence


class CaptureEnvironmentError(RuntimeError):
    """The visible desktop cannot safely produce a current recommendation."""


def _default_screen_grabber() -> np.ndarray:
    """Full-desktop grabber, kept for injected/test compatibility."""
    rgb = np.asarray(ImageGrab.grab(all_screens=False))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _roi_screen_grabber(bbox) -> Callable[[], np.ndarray]:
    """Grab only the recommendation panel ROI instead of the whole desktop.

    Capturing the 271x938 panel directly skips transferring and converting
    the other ~88% of a 1920x1080 frame on every capture.
    """
    def grab() -> np.ndarray:
        rgb = np.asarray(ImageGrab.grab(bbox=bbox, all_screens=False))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return grab


def _default_dpi_reader() -> int:
    try:
        return int(ctypes.windll.user32.GetDpiForSystem())
    except Exception:
        return 0


def _default_process_checker() -> bool:
    try:
        import psutil
        names = {
            (process.info.get("name") or "").lower()
            for process in psutil.process_iter(["name"])
        }
        return "hearthstone.exe" in names and "hsang.exe" in names
    except Exception:
        return False


def _default_window_reader() -> tuple[int, bool]:
    try:
        import win32gui
        from get_screen import get_HS_hwnd, is_allowed_hearthstone_foreground
        hwnd = int(get_HS_hwnd())
        return hwnd, bool(hwnd and is_allowed_hearthstone_foreground(
            hwnd, win32gui.GetForegroundWindow()))
    except Exception:
        return 0, False


class DesktopCapture:
    def __init__(
        self,
        config: RecommendationConfig,
        screen_grabber: Optional[Callable[[], np.ndarray]] = None,
        dpi_reader: Callable[[], int] = _default_dpi_reader,
        process_checker: Callable[[], bool] = _default_process_checker,
        window_reader: Callable[[], tuple[int, bool]] = _default_window_reader,
        clock: Callable[[], float] = time.time,
    ):
        self.config = config
        self.screen_grabber = (screen_grabber
                               or _roi_screen_grabber(
                                   config.recommendation_roi))
        self.dpi_reader = dpi_reader
        self.process_checker = process_checker
        self.window_reader = window_reader
        self.clock = clock
        # 换牌结束后（对战期）的小截图模式：只抓策略推荐区。
        # 换牌期保持 None（全图读取，不碰换牌逻辑）。
        self.strategy_roi = None

    def set_strategy_roi(self, roi: Optional[tuple[int, int, int, int]]) -> None:
        self.strategy_roi = roi

    def clear_strategy_roi(self) -> None:
        self.strategy_roi = None

    @property
    def _roi_size(self) -> tuple[int, int]:
        left, top, right, bottom = self.config.recommendation_roi
        return (bottom - top, right - left)

    def _is_roi_frame(self, width: int, height: int) -> bool:
        roi_height, roi_width = self._roi_size
        return (width, height) == (roi_width, roi_height)

    def from_file(self, path: Path | str) -> FrameEvidence:
        data = np.fromfile(str(path), dtype=np.uint8)
        pixels = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if pixels is None:
            raise CaptureEnvironmentError("frame_decode_failed")
        return self.from_array(pixels, validate_environment=False)

    def capture(self, ocr_panel_ok: bool = False) -> FrameEvidence:
        if not self.process_checker():
            raise CaptureEnvironmentError("required_process_missing")
        hwnd, foreground = self.window_reader()
        if not hwnd:
            raise CaptureEnvironmentError("hearthstone_window_missing")
        if not foreground:
            raise CaptureEnvironmentError("hearthstone_not_foreground")
        if self.strategy_roi is not None:
            return self._capture_strategy_region(hwnd, foreground)
        pixels = self.screen_grabber()
        try:
            return self.from_array(
                pixels, validate_environment=True, dpi=self.dpi_reader(),
                window_handle=hwnd, foreground=foreground)
        except CaptureEnvironmentError as exc:
            if "recommendation_panel_missing" not in str(exc):
                raise
            if not ocr_panel_ok:
                raise
            # 换牌面板的红头判定并不可靠；交由 OCR 证据裁定：
            # 识别出"打法参考A"（stable_reader.required_headers）即确认
            # 面板在，否则由 reader 置零置信度拒绝，安全兜底。
            return self.from_array(
                pixels, validate_environment=False, dpi=self.dpi_reader(),
                window_handle=hwnd, foreground=foreground)

    def _capture_strategy_region(
            self, hwnd: int, foreground: bool) -> FrameEvidence:
        try:
            rgb = np.asarray(ImageGrab.grab(
                bbox=self.strategy_roi, all_screens=False))
        except Exception:
            self.strategy_roi = None
            raise CaptureEnvironmentError("recommendation_panel_missing")
        pixels = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        roi = pixels[:, :, :3]
        if roi.size == 0 or float(np.std(roi)) < 8.0:
            # 子区域无可信内容（面板可能换了布局/消失）→ 回退全图
            self.strategy_roi = None
            raise CaptureEnvironmentError("recommendation_panel_missing")
        return FrameEvidence(
            frame_id=f"frame-{uuid.uuid4()}", captured_at=self.clock(),
            desktop_size=self.config.desktop_size,
            dpi=self.config.desktop_dpi,
            window_handle=hwnd, foreground=foreground,
            recommendation_roi=self.strategy_roi,
            exact_hash=hashlib.sha256(roi.tobytes()).hexdigest(),
            perceptual_hash=self._perceptual_hash(roi),
            panel_visible=True, pixels=roi)

    def from_array(
        self,
        pixels: np.ndarray,
        *,
        validate_environment: bool,
        dpi: Optional[int] = None,
        window_handle: int = 0,
        foreground: bool = True,
    ) -> FrameEvidence:
        if pixels is None or pixels.ndim != 3 or pixels.shape[2] < 3:
            raise CaptureEnvironmentError("invalid_frame")
        height, width = pixels.shape[:2]
        roi_frame = self._is_roi_frame(width, height)
        if validate_environment:
            if roi_frame:
                # ROI grabber output: the desktop size is known from config.
                desktop_width, desktop_height = self.config.desktop_size
            else:
                desktop_width, desktop_height = width, height
                if (width, height) != self.config.desktop_size:
                    raise CaptureEnvironmentError(
                        f"desktop_size_mismatch:{width}x{height}")
        else:
            desktop_width, desktop_height = width, height
        effective_dpi = self.config.desktop_dpi if dpi is None else int(dpi)
        if validate_environment and effective_dpi != self.config.desktop_dpi:
            raise CaptureEnvironmentError(f"dpi_mismatch:{effective_dpi}")

        roi = (pixels[:, :, :3] if roi_frame
               else self._crop_array(pixels))
        panel_visible = self._panel_is_visible(roi)
        if validate_environment and not panel_visible:
            raise CaptureEnvironmentError("recommendation_panel_missing")
        exact_hash = hashlib.sha256(roi.tobytes()).hexdigest()
        perceptual_hash = self._perceptual_hash(roi)
        return FrameEvidence(
            frame_id=f"frame-{uuid.uuid4()}", captured_at=self.clock(),
            desktop_size=(desktop_width, desktop_height), dpi=effective_dpi,
            window_handle=window_handle, foreground=foreground,
            recommendation_roi=self.config.recommendation_roi,
            exact_hash=exact_hash, perceptual_hash=perceptual_hash,
            panel_visible=panel_visible, pixels=pixels,
        )

    def crop_recommendation(self, frame: FrameEvidence) -> np.ndarray:
        height, width = frame.pixels.shape[:2]
        if self._is_roi_frame(width, height):
            return frame.pixels[:, :, :3].copy()
        if (self.strategy_roi is not None
                and frame.recommendation_roi == self.strategy_roi):
            return frame.pixels[:, :, :3].copy()
        return self._crop_array(frame.pixels).copy()

    def _crop_array(self, pixels: np.ndarray) -> np.ndarray:
        left, top, right, bottom = self.config.recommendation_roi
        height, width = pixels.shape[:2]
        if left < 0 or top < 0 or right > width or bottom > height:
            raise CaptureEnvironmentError("recommendation_roi_out_of_bounds")
        return pixels[top:bottom, left:right, :3]

    @staticmethod
    def _panel_is_visible(roi: np.ndarray) -> bool:
        if roi.size == 0 or float(np.std(roi)) < 8.0:
            return False
        # HSAng uses a dark red header at the top of this fixed panel.
        header = roi[:45]
        b, g, r = cv2.split(header)
        red_dark = (r > g * 1.25) & (r > b * 1.15) & (r < 140)
        return float(np.mean(red_dark)) > 0.08

    @staticmethod
    def _perceptual_hash(roi: np.ndarray) -> str:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA)
        bits = resized >= float(np.mean(resized))
        return "".join(f"{value:02x}" for value in np.packbits(bits))
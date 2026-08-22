"""Lazy PaddleOCR boundary with normalized evidence output."""

import cv2
import importlib.machinery
import importlib.util
import os
from pathlib import Path
import sys
import time

from src.recommendation_models import OcrEvidence, OcrLine


class OcrUnavailableError(RuntimeError):
    pass


def _recommended_threads() -> int:
    """按机器实际情况推荐 OpenMP/MKL 线程数。

    优先取物理核数（超线程对 paddle CPU 推理基本无收益，用满会与
    游戏/盒子抢核）；无法取到时按逻辑核折半，最低 4 线程。
    """
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        if physical:
            return max(4, int(physical))
    except Exception:
        pass
    logical = os.cpu_count() or 4
    return max(4, logical // 2)


def _external_click_module():
    """Load PyPI's click package instead of this project's click.py."""
    project_root = os.path.normcase(str(Path(__file__).resolve().parents[2]))
    search_path = [
        entry for entry in sys.path
        if entry and os.path.normcase(os.path.abspath(entry)) != project_root
    ]
    spec = importlib.machinery.PathFinder.find_spec("click", search_path)
    if spec is None or spec.loader is None or not spec.origin:
        raise OcrUnavailableError("external click package not found")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get("click")
    sys.modules["click"] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if previous is not None:
            sys.modules["click"] = previous
        else:
            sys.modules.pop("click", None)
    return module


class PaddleOcrAdapter:
    name = "paddleocr"

    def __init__(self, engine=None, clock=time.time, engine_factory=None,
                 model_root=None):
        self.engine = engine
        self.clock = clock
        self.engine_factory = engine_factory
        # 每次 OCR 的输入图按顺序存盘（会话子目录，便于调试查看到底
        # 识别了多大的图）。默认关闭；设置 OCR_FRAME_DIR=<目录> 开启。
        frame_root = os.environ.get("OCR_FRAME_DIR", "")
        self._frame_counter = 0
        self._frame_dir = None
        if frame_root and frame_root != "0":
            run_dir = Path(frame_root) / time.strftime("run_%H%M%S")
            run_dir.mkdir(parents=True, exist_ok=True)
            self._frame_dir = run_dir
            print(f"[OCR] 本会话自动保存截图: {run_dir}")
        default_root = (Path(os.environ.get("LOCALAPPDATA", Path.home()))
                        / "AutoHS" / "ocr_models" / "paddleocr")
        self.model_root = Path(model_root or default_root).resolve()

    def load(self):
        if self.engine is not None:
            return self
        try:
            # Multi-threaded OpenMP inference (measured 4.5x faster than the
            # single-threaded default on a 16-core machine). Threads adapt to
            # the machine's physical core count; use setdefault so an
            # explicit user override still wins.
            threads = str(_recommended_threads())
            os.environ.setdefault("OMP_NUM_THREADS", threads)
            os.environ.setdefault("MKL_NUM_THREADS", threads)
            print(f"[OCR] 推理线程 OMP={os.environ['OMP_NUM_THREADS']}")
            project_click = sys.modules.get("click")
            sys.modules["click"] = _external_click_module()
            if self.engine_factory is None:
                from paddleocr import PaddleOCR
                self.engine_factory = PaddleOCR
            self.model_root.mkdir(parents=True, exist_ok=True)
            self.engine = self.engine_factory(
                use_angle_cls=False, lang="ch", show_log=False,
                det_model_dir=str(self.model_root / "det_ch_ppocrv4"),
                rec_model_dir=str(self.model_root / "rec_ch_ppocrv4"),
                cls_model_dir=str(self.model_root / "cls_ch_mobile_v2"),
            )
        except Exception as exc:
            raise OcrUnavailableError(
                f"PaddleOCR unavailable: {type(exc).__name__}: {exc}") from exc
        finally:
            if "project_click" in locals() and project_click is not None:
                sys.modules["click"] = project_click
            elif "project_click" in locals():
                sys.modules.pop("click", None)
        return self

    def recognize(self, image, frame_id, preprocessing):
        self.load()
        started = time.perf_counter()
        try:
            raw = self.engine.ocr(image, cls=False)
        except Exception as exc:
            raise OcrUnavailableError(
                f"PaddleOCR recognition failed: {type(exc).__name__}: {exc}") from exc
        lines = []
        for page in raw or []:
            for item in page or []:
                box, pair = item
                text, confidence = pair
                lines.append(OcrLine(
                    str(text).strip(), float(confidence),
                    tuple((float(x), float(y)) for x, y in box)))
        lines.sort(key=lambda item: min(
            (point[1] for point in item.box), default=0.0))
        normalized = "\n".join(line.text for line in lines if line.text)
        confidence = min((line.confidence for line in lines), default=0.0)
        height, width = image.shape[:2] if image is not None else (0, 0)
        print(f"[OCR] 图 {width}x{height} 行 {len(lines)} 置信 {confidence:.2f} "
              f"耗时 {(time.perf_counter() - started) * 1000:.0f}ms "
              f"({preprocessing})")
        if self._frame_dir is not None:
            self._frame_counter += 1
            safe_name = str(preprocessing or "frame").replace("/", "_")
            cv2.imwrite(
                str(self._frame_dir / f"{self._frame_counter:03d}_{safe_name}.png"),
                image)
        return OcrEvidence(
            frame_id, self.clock(), tuple(lines), normalized, confidence,
            self.name, preprocessing)

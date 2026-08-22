"""Conservative configuration for the first recommendation automation release."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RecommendationConfig:
    desktop_size: tuple[int, int] = (1920, 1080)
    desktop_dpi: int = 96
    recommendation_roi: tuple[int, int, int, int] = (7, 32, 278, 970)
    # 对战期小截图模式（换牌结束后切换；换牌期不使用）：
    # 屏幕坐标 (0,0) 为原点，左上 (0,230)，宽 225（250x0.9），
    # 高 205（240-35）。
    strategy_roi: tuple[int, int, int, int] = (0, 230, 225, 435)
    max_attempts: int = 3
    stable_frames: int = 2
    min_ocr_confidence: float = 0.70
    retry_interval_seconds: float = 0.1
    # 游戏开始第 7 秒才开始换牌识图（盒子面板此时已就位）。
    mulligan_ready_delay_seconds: float = 7.0
    post_action_delay_seconds: float = 0.0
    # 每个新回合开始时延时一次（给盒子更新推荐留时间），同回合操作间不重复延时。
    pre_action_delay_seconds: float = 2.0
    recognition_timeout_seconds: float = 2.0
    result_timeout_seconds: float = 5.0
"""Execute exactly one validated recommendation and verify its result."""

from dataclasses import dataclass
from enum import Enum
import time

from src.recommendation_models import ActionKind
from src.safety.recommendation_validator import ConsumedActionStore


class FlowStepStatus(str, Enum):
    EXECUTED = "executed"
    OBSERVE = "observe"
    RETRY = "retry"


class ResultStatus(str, Enum):
    COMPLETED = "completed"
    WAITING_RECOMMENDATION = "waiting_recommendation"
    RETRY = "retry"


@dataclass(frozen=True)
class FlowStepResult:
    status: FlowStepStatus
    diagnostics: str = ""


class RecommendationFlow:
    def __init__(self, capture, reader, parser, state_supplier, adapter,
                 validator, controller, sleep=time.sleep, clock=time.time,
                 result_timeout=5.0, stopped=lambda: False,
                 consumed=None, post_action_delay=0.0):
        self.capture = capture
        self.reader = reader
        self.parser = parser
        self.state_supplier = state_supplier
        self.adapter = adapter
        self.validator = validator
        self.controller = controller
        self.sleep = sleep
        self.clock = clock
        # Kept for constructor compatibility; result verification no longer
        # waits for a timeout window, it checks once and retries immediately.
        self.result_timeout = result_timeout
        self.stopped = stopped
        self.consumed = consumed or ConsumedActionStore()
        self.post_action_delay = post_action_delay
        self.waiting_instruction = None
        self.waiting_panel_hash = None
        self.waiting_turn_number = None
        self.waiting_action_kind = None

    def run_player_turn_step(self):
        try:
            state, revision = self.state_supplier()
            if not state.is_my_turn:
                return FlowStepResult(FlowStepStatus.OBSERVE, "opponent_turn")
            if (self.waiting_turn_number is not None
                    and self.waiting_turn_number
                    != state.game_num_turns_in_play):
                self._clear_waiting()
            observed = {}
            def frame_supplier():
                observed["frame"] = self.capture.capture()
                return observed["frame"]
            evidence = self.reader.read(
                frame_supplier, self.capture.crop_recommendation)
            stable_frame = observed["frame"]
            if self.waiting_instruction is not None:
                same_instruction = (
                    evidence.normalized_text == self.waiting_instruction)
                if (same_instruction
                        and self.waiting_action_kind
                        == ActionKind.CHOOSE_DISCOVER):
                    if getattr(state, "discover_choice_count", None) in (
                            1, 2, 3):
                        self._clear_waiting()
                        return FlowStepResult(
                            FlowStepStatus.RETRY,
                            "discover_choice_still_open")
                    return FlowStepResult(
                        FlowStepStatus.OBSERVE,
                        "waiting_recommendation_update")
                same_panel = self._hash_distance(
                    stable_frame.perceptual_hash,
                    self.waiting_panel_hash) <= 4
                if same_instruction and same_panel:
                    return FlowStepResult(
                        FlowStepStatus.OBSERVE,
                        "waiting_recommendation_update")
                self._clear_waiting()
            proposed = self.parser.parse(
                evidence, state.game_num_turns_in_play, revision)
            if proposed.action == ActionKind.MULLIGAN:
                return FlowStepResult(
                    FlowStepStatus.OBSERVE,
                    "stale_mulligan_recommendation")
            adapted = self.adapter(proposed, state)
            fresh_state, fresh_revision = self.state_supplier()
            current_frame = self.capture.capture()
            current_evidence = self.reader.read_frame(
                current_frame, self.capture.crop_recommendation)
            if (current_evidence.normalized_text
                    != evidence.normalized_text):
                return FlowStepResult(
                    FlowStepStatus.RETRY, "recommendation_changed")
            validation = self.validator.validate(
                proposed, adapted, stable_frame, current_frame,
                current_evidence,
                fresh_state, fresh_revision, "PLAYER_TURN", self.consumed,
                self.stopped())
            if not validation.accepted:
                return FlowStepResult(FlowStepStatus.RETRY, validation.code)
            adapted, key = validation.value
            print(f"[推荐] {proposed.normalized_instruction}")
            print("[执行] 开始点击。")
            result = self.controller.execute(adapted.manual_action, fresh_state)
            if not result.executed or result.recovery_needed:
                return FlowStepResult(FlowStepStatus.RETRY,
                                      "execution_failed")
            # 操作结束后延时再开始下一轮截图+OCR（盒子更新面板留时间）。
            self.sleep(self.post_action_delay)
            verified = self._verify_result(
                proposed, adapted, current_frame,
                fresh_state, fresh_revision)
            if verified == ResultStatus.COMPLETED:
                self.consumed.mark_consumed(key)
                return FlowStepResult(FlowStepStatus.EXECUTED)
            if verified == ResultStatus.WAITING_RECOMMENDATION:
                # A discover click is not complete while Power.log still
                # exposes a choice UI.  Keep it retryable until that UI
                # closes; all other confirmed actions retain duplicate
                # execution protection.
                if proposed.action != ActionKind.CHOOSE_DISCOVER:
                    self.consumed.mark_consumed(key)
                self.waiting_instruction = proposed.normalized_instruction
                self.waiting_panel_hash = current_frame.perceptual_hash
                self.waiting_turn_number = proposed.turn_number
                self.waiting_action_kind = proposed.action
                return FlowStepResult(
                    FlowStepStatus.OBSERVE,
                    "waiting_recommendation_update")
            return FlowStepResult(
                FlowStepStatus.RETRY, "result_not_confirmed")
        except Exception as exc:
            return FlowStepResult(
                FlowStepStatus.RETRY,
                f"{type(exc).__name__}:{exc}")

    def _frame_for(self, frame_id):
        # Capture implementations may retain the stable reader's current frame.
        current = getattr(self.capture, "last_frame", None)
        if current is not None and current.frame_id == frame_id:
            return current
        # Tests and simple suppliers can provide a fresh identity-equivalent frame.
        return type("FrameProxy", (), {
            "frame_id": frame_id, "desktop_size": (1920, 1080), "dpi": 96,
            "window_handle": 1, "foreground": True, "panel_visible": True,
        })()

    def _verify_result(self, proposed, adapted, before_frame, before_state,
                       before_revision):
        """Check the result once right after executing; no timeout waiting.

        A changed recommendation text confirms completion. A discover
        action or a satisfied log postcondition waits for the next
        recommendation. Anything else returns RETRY immediately so the
        caller re-executes the same action with fresh evidence instead of
        waiting for a result window.
        """
        before_instruction = proposed.normalized_instruction
        after_state, after_revision = self.state_supplier()
        log_changed = (after_revision > before_revision
                       and self._postcondition(
                           adapted.postcondition, before_state, after_state,
                           adapted.source_entity_id,
                           adapted.target_entity_id))
        recommendation_changed = False
        try:
            after_frame = self.capture.capture()
            if after_frame.exact_hash != before_frame.exact_hash:
                supplied = False
                def frame_supplier():
                    nonlocal supplied
                    if not supplied:
                        supplied = True
                        return after_frame
                    return self.capture.capture()
                after_evidence = self.reader.read(
                    frame_supplier, self.capture.crop_recommendation)
                recommendation_changed = (
                    after_evidence.normalized_text != before_instruction)
        except Exception:
            recommendation_changed = False
        if recommendation_changed:
            return ResultStatus.COMPLETED
        if proposed.action == ActionKind.CHOOSE_DISCOVER:
            return ResultStatus.WAITING_RECOMMENDATION
        if log_changed:
            return ResultStatus.WAITING_RECOMMENDATION
        return ResultStatus.RETRY

    def _clear_waiting(self):
        self.waiting_instruction = None
        self.waiting_panel_hash = None
        self.waiting_turn_number = None
        self.waiting_action_kind = None

    @staticmethod
    def _hash_distance(left, right):
        if not left or not right or len(left) != len(right):
            return float("inf")
        try:
            return sum(
                (int(a, 16) ^ int(b, 16)).bit_count()
                for a, b in zip(left, right))
        except ValueError:
            return 0 if left == right else float("inf")

    @staticmethod
    def _postcondition(kind, before, after, source_entity_id=None,
                       target_entity_id=None):
        if kind == "turn_changed":
            return (after.game_num_turns_in_play != before.game_num_turns_in_play
                    or not after.is_my_turn)
        if kind == "hand_card_left":
            return source_entity_id not in {
                getattr(card, "entity_id", None)
                for card in after.my_hand_cards}
        if kind == "hero_power_changed":
            return (getattr(after.my_hero_power, "exhausted", 0)
                    != getattr(before.my_hero_power, "exhausted", 0)
                    or after.my_last_mana != before.my_last_mana)
        if kind == "choice_resolved":
            if (getattr(before, "discover_choice_count", None) in (1, 2, 3)
                    and getattr(after, "discover_choice_count", None) is None):
                return True
            before_hand = tuple(
                getattr(card, "entity_id", None)
                for card in before.my_hand_cards)
            after_hand = tuple(
                getattr(card, "entity_id", None)
                for card in after.my_hand_cards)
            return before_hand != after_hand
        if kind == "location_changed":
            old = RecommendationFlow._entity_by_id(
                before.my_locations, source_entity_id)
            new = RecommendationFlow._entity_by_id(
                after.my_locations, source_entity_id)
            return new is None or (
                getattr(new, "action_cooldown", 0)
                != getattr(old, "action_cooldown", 0)
                or getattr(new, "damage", 0) != getattr(old, "damage", 0))
        if kind == "starship_launched":
            old = RecommendationFlow._entity_by_id(
                before.my_minions, source_entity_id)
            new = RecommendationFlow._entity_by_id(
                after.my_minions, source_entity_id)
            return old is not None and (
                new is None
                or getattr(new, "card_id", None)
                != getattr(old, "card_id", None))
        if kind == "combat_state_changed":
            old_source = RecommendationFlow._entity_by_id(
                before.my_minions, source_entity_id)
            new_source = RecommendationFlow._entity_by_id(
                after.my_minions, source_entity_id)
            old_target = RecommendationFlow._entity_by_id(
                before.oppo_minions, target_entity_id)
            new_target = RecommendationFlow._entity_by_id(
                after.oppo_minions, target_entity_id)
            return (new_source is None or new_target is None
                    or getattr(new_source, "exhausted", 0)
                    != getattr(old_source, "exhausted", 0)
                    or getattr(new_target, "damage", 0)
                    != getattr(old_target, "damage", 0))
        if kind == "hero_combat_state_changed":
            old_hero = getattr(before, "my_hero", None)
            new_hero = getattr(after, "my_hero", None)
            old_target = RecommendationFlow._entity_by_id(
                before.oppo_minions, target_entity_id)
            new_target = RecommendationFlow._entity_by_id(
                after.oppo_minions, target_entity_id)
            return (new_hero is None or new_target is None
                    or getattr(new_hero, "exhausted", 0)
                    != getattr(old_hero, "exhausted", 0)
                    or getattr(new_hero, "damage", 0)
                    != getattr(old_hero, "damage", 0)
                    or getattr(new_target, "damage", 0)
                    != getattr(old_target, "damage", 0))
        return False

    @staticmethod
    def _entity_by_id(collection, entity_id):
        return next((entity for entity in collection
                     if getattr(entity, "entity_id", None) == entity_id), None)
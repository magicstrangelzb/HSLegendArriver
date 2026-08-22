"""Execute all validated mulligan slots and confirm immediately."""

from dataclasses import dataclass
from enum import Enum
import time


class MulliganStatus(str, Enum):
    CONFIRMED = "confirmed"
    CONCEDE = "concede"


@dataclass(frozen=True)
class MulliganResult:
    status: MulliganStatus
    selected_zero_based_slots: tuple[int, ...] = ()
    diagnostics: str = ""


class MulliganFlow:
    def __init__(self, executor, action_supplier, state_supplier,
                 action_context=None, stopped=lambda: False,
                 sleep=time.sleep, pre_action_delay=5.0):
        self.executor = executor
        self.action_supplier = action_supplier
        self.state_supplier = state_supplier
        self.stopped = stopped
        self.sleep = sleep
        self.pre_action_delay = pre_action_delay
        if action_context is None:
            from contextlib import nullcontext
            action_context = nullcontext
        self.action_context = action_context

    def run(self):
        try:
            initial = self.state_supplier()
            action = self.action_supplier()
            current_action = self.action_supplier()
            fresh = self.state_supplier()
            if self.stopped():
                return MulliganResult(MulliganStatus.CONCEDE,
                                      diagnostics="stopped")
            if self._identity(initial) != self._identity(fresh):
                return MulliganResult(MulliganStatus.CONCEDE,
                                      diagnostics="hand_changed")
            if (getattr(initial, "log_revision", None)
                    != getattr(fresh, "log_revision", None)):
                return MulliganResult(MulliganStatus.CONCEDE,
                                      diagnostics="revision_changed")
            if (getattr(fresh, "is_end", False)
                    or getattr(fresh, "game_num_turns_in_play", 0) != 0):
                return MulliganResult(MulliganStatus.CONCEDE,
                                      diagnostics="mulligan_stage_changed")
            if (current_action.normalized_instruction
                    != action.normalized_instruction
                    or current_action.mulligan_slots != action.mulligan_slots):
                return MulliganResult(MulliganStatus.CONCEDE,
                                      diagnostics="recommendation_changed")
            count = len(fresh.my_hand_cards)
            selected = tuple(sorted({slot - 1 for slot in action.mulligan_slots}))
            clickable_count = {3: 3, 5: 4}.get(count, 0)
            if count not in (3, 5) or any(
                    index < 0 or index >= clickable_count
                    for index in selected):
                return MulliganResult(MulliganStatus.CONCEDE,
                                      diagnostics="mulligan_slot_invalid")
            print(f"已识别换牌建议，等待 {self.pre_action_delay:.0f}s 后执行……")
            self.sleep(self.pre_action_delay)
            with self.action_context():
                for index in selected:
                    self.executor.replace_starting_card(index, count)
                self.executor.commit_choose_card()
            return MulliganResult(MulliganStatus.CONFIRMED, selected)
        except Exception as exc:
            try:
                self.executor.cancel_click()
            except Exception:
                pass
            return MulliganResult(
                MulliganStatus.CONCEDE,
                diagnostics=f"{type(exc).__name__}:{exc}")

    @staticmethod
    def _identity(state):
        return tuple((card.card_id, getattr(card, "entity_id", None))
                     for card in state.my_hand_cards)
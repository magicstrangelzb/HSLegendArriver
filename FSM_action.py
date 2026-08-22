import _thread
import random
import sys
import threading
import time

import keyboard

import requests

import click
import get_screen
from manual_controller import (
    ClickExecutor, GlobalHotkeyInput, ManualController,
)
from strategy import StrategyState
from log_state import *
from src.capture.desktop_capture import DesktopCapture
from src.flow.mulligan_flow import MulliganFlow, MulliganStatus
from src.flow.recommendation_flow import (
    FlowStepStatus, RecommendationFlow,
)
from src.game_state.recommendation_adapter import adapt_action
from src.ocr.paddle_adapter import PaddleOcrAdapter
from src.ocr.stable_reader import StableRecommendationReader
from src.parser.recommendation_parser import RecommendationParser
from src.recommendation_config import RecommendationConfig
from src.recommendation_models import ActionKind
from src.safety.recommendation_validator import RecommendationValidator

from datetime import datetime

FSM_state = ""
time_begin = 0.0
game_count = 0
win_count = 0
quitting_flag = False
# 定时计划 / Web 控制台：置 True 表示“本局对战结束后停止自动化”。
# 对局进行中（换牌/对战/结算）不会立即退出，只有回到非对局状态才停止。
stop_after_current_game = False
shutdown_event = threading.Event()
log_state = LogState()
log_iter = log_iter_func(HEARTHSTONE_LOG_ROOT)
choose_hero_count = 0
manual_controller = ManualController(
    input_func=GlobalHotkeyInput(
        keyboard, shutdown_event=shutdown_event),
    executor=ClickExecutor(click),
)
auto_mulligan_flow = None
recommendation_flow = None
recommendation_config = None
recommendation_capture = None
recommendation_parser = None
recommendation_reader = None
recommendation_validator = None
active_game_generation = -1
mulligan_delay_generation = None
player_turn_delay_key = None
last_automation_diagnostic = None
_snapshot_cache_key = None
_snapshot_cache = None
# 调试快照写盘节流：日志每次变化都全量序列化整个 log_state 会拖慢主循环，
# 只在间隔 SNAPSHOT_WRITE_INTERVAL 秒后重新写盘。
SNAPSHOT_WRITE_INTERVAL = 5.0
_last_snapshot_write = 0.0


def _automation_state():
    snapshot = refresh_snapshot()
    if snapshot is None:
        raise RuntimeError("power_log_snapshot_unavailable")
    snapshot.log_revision = log_state.revision
    return snapshot


def _automation_state_with_revision():
    snapshot = _automation_state()
    return snapshot, log_state.revision


def initialize_recommendation_automation():
    """Rebuild per-game flows while reusing expensive OCR components."""
    global auto_mulligan_flow, recommendation_flow
    global recommendation_config, recommendation_capture
    global recommendation_parser, recommendation_reader
    global recommendation_validator

    if recommendation_config is None:
        recommendation_config = RecommendationConfig()
        recommendation_capture = DesktopCapture(recommendation_config)
        recommendation_parser = RecommendationParser()
        recommendation_reader = StableRecommendationReader(
            recommendation_config, PaddleOcrAdapter(),
            text_normalizer=recommendation_parser.normalize_action_text,
            # 固定策略区 (0,230)-(261,630) 内含"打法参考A"标题，
            # 仍用作面板信标（配合 parser 指令句式双重把关）。
            required_headers=("打法参考A", "打法参考Ａ"))
        recommendation_validator = RecommendationValidator(
            recommendation_config)

    def read_mulligan_action():
        # 换牌面板是否在，由 OCR 证据裁定："打法参考A"标题被识别到
        # 即承认面板存在（required_headers 过滤 + 换牌句式校验兜底）。
        evidence = recommendation_reader.read(
            lambda: recommendation_capture.capture(ocr_panel_ok=True),
            recommendation_capture.crop_recommendation)
        action = recommendation_parser.parse(
            evidence, log_state.game_num_turns_in_play, log_state.revision)
        if action.action != ActionKind.MULLIGAN:
            raise RuntimeError("recommendation_is_not_mulligan")
        return action

    auto_mulligan_flow = MulliganFlow(
        click, read_mulligan_action, _automation_state,
        action_context=click.hearthstone_action_session,
        stopped=shutdown_event.is_set)
    recommendation_flow = RecommendationFlow(
        capture=recommendation_capture,
        reader=recommendation_reader,
        parser=recommendation_parser,
        state_supplier=_automation_state_with_revision,
        adapter=adapt_action,
        validator=recommendation_validator,
        controller=manual_controller,
        result_timeout=recommendation_config.result_timeout_seconds,
        post_action_delay=recommendation_config.post_action_delay_seconds,
        stopped=shutdown_event.is_set,
    )


def reset_game_session():
    """Clear every match-scoped automation state for a newly created game."""
    global active_game_generation, choose_hero_count
    global mulligan_delay_generation, player_turn_delay_key
    global last_automation_diagnostic
    global _snapshot_cache_key, _snapshot_cache
    initialize_recommendation_automation()
    active_game_generation = log_state.game_generation
    choose_hero_count = 0
    mulligan_delay_generation = None
    player_turn_delay_key = None
    last_automation_diagnostic = None
    _snapshot_cache_key = None
    _snapshot_cache = None
    click.center_mouse()


def init():
    global log_state, log_iter, choose_hero_count, active_game_generation
    global mulligan_delay_generation, player_turn_delay_key
    global last_automation_diagnostic
    global _snapshot_cache_key, _snapshot_cache

    log_state = LogState()
    log_iter = log_iter_func(HEARTHSTONE_LOG_ROOT)
    choose_hero_count = 0
    active_game_generation = -1
    mulligan_delay_generation = None
    player_turn_delay_key = None
    last_automation_diagnostic = None
    _snapshot_cache_key = None
    _snapshot_cache = None
    shutdown_event.clear()
    initialize_recommendation_automation()
    click.center_mouse()


def update_log_state():
    global active_game_generation
    global _last_snapshot_write
    log_container = next(log_iter)
    if log_container.log_type == LOG_CONTAINER_ERROR:
        return False

    previous_revision = log_state.revision
    for log_line_container in log_container.message_list:
        ok = update_state(log_state, log_line_container)
        # if not ok:
        #     return False

    if log_state.game_generation != active_game_generation:
        reset_game_session()

    if (DEBUG_FILE_WRITE and log_state.revision != previous_revision
            and time.time() - _last_snapshot_write
            >= SNAPSHOT_WRITE_INTERVAL):
        _last_snapshot_write = time.time()
        with open("./log/game_state_snapshot.txt", "w", encoding="utf8") as f:
            f.write(str(log_state))

    # 注意如果Power.log没有更新, 这个函数依然会返回. 应该考虑到game_state只是被初始化
    # 过而没有进一步更新的可能
    if log_state.game_entity_id == 0:
        return False

    return True


def refresh_snapshot():
    """Read pending Power.log events and build a fresh manual snapshot."""
    global _snapshot_cache_key, _snapshot_cache
    if not update_log_state():
        return None
    cache_key = (log_state.game_generation, log_state.revision)
    if cache_key != _snapshot_cache_key:
        _snapshot_cache = StrategyState(log_state)
        _snapshot_cache_key = cache_key
    return _snapshot_cache


def wait_for_log_update(start_revision=None, timeout=2.0):
    """Wait briefly for evidence that an input changed game state."""
    if start_revision is None:
        start_revision = log_state.revision
    deadline = time.time() + timeout
    while time.time() < deadline:
        if update_log_state() and log_state.revision > start_revision:
            return True
    manual_controller.output("尚未检测到状态变化，请查看游戏后刷新或重试。")
    return False


def wait_until_battle_starts():
    loop_count = 0
    while True:
        if not update_log_state():
            return FSM_ERROR
        if log_state.is_end:
            return FSM_QUITTING_BATTLE
        if log_state.game_num_turns_in_play > 0:
            return FSM_BATTLING
        loop_count += 1
        if loop_count >= 60:
            warn_print("Time out in Choosing Card")
            return FSM_ERROR
        time.sleep(STATE_CHECK_INTERVAL)


def push_wx(sckey, desp=""):
    """
    推送消息到微信
    """
    if sckey == '':
        print("[注意] 未提供sckey，不进行推送！")
    else:
        server_url = f"https://sc.ftqq.com/{sckey}.send"
        params = {
            "text": '小米运动 步数修改',
            "desp": desp
        }

        response = requests.get(server_url, params=params)
        json_data = response.json()

        if json_data['data']['errno'] == 0:
            print("{}, {}点，推送成功".format(datetime.now().date(), datetime.now().hour))
        else:
            print("{}, {}点，推送失败，{}，{}".format(datetime.now().date(), datetime.now().hour, json_data['data']['errno'],
                                              json_data['data']['errmsg']))
            # print(f"[{datetime.now().date()}] 推送失败：{json_data['data']['errno']}({json_data['data']['errmsg']})")


def system_exit():
    global quitting_flag

    sckey = ""
    # 这个码是推送信息到微信的码，去sct.ftqq.com上绑定，不需要就不用管

    push = f"一共完成了{game_count}场对战, 赢了{win_count}场"

    try:
        push_wx(sckey, push)
    except:
        print("无法推送")

    sys_print(f"一共完成了{game_count}场对战, 赢了{win_count}场")
    print_info_close()

    quitting_flag = True
    shutdown_event.set()
    if threading.current_thread() is threading.main_thread():
        raise SystemExit(0)
    _thread.interrupt_main()


def request_stop_after_game():
    """请求“本局对战结束后停止”。

    对局进行中时不会中断当前操作；当状态机回到非对局状态
    （主菜单/选职业/匹配/炉石未运行等）后自动化线程自动退出。
    """
    global stop_after_current_game
    stop_after_current_game = True
    info_print("已请求：本局对战结束后停止自动化。")
    return True


def request_immediate_stop():
    """Web 模式下的立即停止：只终止自动化线程，不影响服务器主线程。"""
    global quitting_flag
    info_print("收到立即停止指令，正在终止自动化……")
    quitting_flag = True
    shutdown_event.set()
    return True


def print_out():
    global FSM_state
    global time_begin
    global game_count

    # sys_print("Enter State " + str(FSM_state))

    if FSM_state == FSM_LEAVE_HS:
        warn_print("HearthStone not found! Try to go back to HS")

    if FSM_state == FSM_CHOOSING_CARD:
        game_count += 1
        # sys_print("The " + str(game_count) + " game begins")
        time_begin = time.time()

    if FSM_state == FSM_QUITTING_BATTLE:
        # sys_print("The " + str(game_count) + " game ends")
        time_now = time.time()
        if time_begin > 0:
            info_print("The last game last for : {} mins {} secs"
                       .format(int((time_now - time_begin) // 60),
                               int(time_now - time_begin) % 60))

    return


def ChoosingHeroAction():
    global choose_hero_count

    if quitting_flag or stop_after_current_game:
        sys.exit(0)

    print_out()

    # 有时脚本会卡在某个地方, 从而在FSM_Matching
    # 和FSM_CHOOSING_HERO之间反复横跳. 这时候要
    # 重启炉石
    # choose_hero_count会在每一次开始留牌时重置
    choose_hero_count += 1
    if choose_hero_count >= 20:
        return FSM_ERROR

    time.sleep(2)
    click.run_hearthstone_action(click.match_opponent)
    time.sleep(1)
    return FSM_MATCHING


def MatchingAction():
    print_out()
    loop_count = 0

    while True:
        if quitting_flag or stop_after_current_game:
            sys.exit(0)

        time.sleep(STATE_CHECK_INTERVAL+random.random()+random.random()+random.random())

        click.run_hearthstone_action(click.commit_error_report)

        ok = update_log_state()
        if ok:
            if not log_state.is_end:
                return FSM_CHOOSING_CARD

        curr_state = get_screen.get_state()
        if curr_state == FSM_CHOOSING_HERO:
            return FSM_CHOOSING_HERO

        loop_count += 1
        # print("寻找对手计时器")
        # print(loop_count)
        if loop_count >= 60:
            warn_print("Time out in Matching Opponent")
            return FSM_ERROR


def ChoosingCardAction():
    global choose_hero_count, mulligan_delay_generation
    choose_hero_count = 0

    print_out()
    snapshot = refresh_snapshot()
    if snapshot is None:
        return FSM_ERROR
    if snapshot.is_end:
        return FSM_QUITTING_BATTLE
    if snapshot.game_num_turns_in_play > 0:
        return FSM_BATTLING

    # CREATE_GAME increments game_generation.  Bind the ready delay to that
    # generation so every match waits once, including matches after the first.
    while mulligan_delay_generation != log_state.game_generation:
        waiting_generation = log_state.game_generation
        delay = (recommendation_config.mulligan_ready_delay_seconds
                 if auto_mulligan_flow is not None else 2)
        time.sleep(delay)
        mulligan_delay_generation = waiting_generation
        snapshot = refresh_snapshot()
        if snapshot is None:
            return FSM_ERROR
        if snapshot.is_end:
            return FSM_QUITTING_BATTLE
        if snapshot.game_num_turns_in_play > 0:
            return FSM_BATTLING

    if auto_mulligan_flow is not None:
        # 重试机制同原始代码：识别失败/执行失败 → 打印诊断 → 返回
        # FSM_CHOOSING_CARD，由外层对战主循环立即重试（~0.2s 一次）。
        result = auto_mulligan_flow.run()
        if result.status == MulliganStatus.CONFIRMED:
            return wait_until_battle_starts()
        manual_controller.output(
            f"换牌推荐暂不可执行，继续重试：{result.diagnostics}")
        return FSM_CHOOSING_CARD

    selected = manual_controller.choose_mulligan(snapshot)
    fresh_snapshot = refresh_snapshot()
    if fresh_snapshot is None:
        return FSM_ERROR
    if not manual_controller.mulligan_is_current(snapshot, fresh_snapshot):
        manual_controller.output("留牌状态已经变化，本次选择未点击，请重新确认。")
        if fresh_snapshot.is_end:
            return FSM_QUITTING_BATTLE
        if fresh_snapshot.game_num_turns_in_play > 0:
            return FSM_BATTLING
        return FSM_CHOOSING_CARD
    try:
        with click.hearthstone_action_session():
            try:
                for hand_index in selected:
                    click.replace_starting_card(
                        hand_index, fresh_snapshot.my_hand_card_num)
                click.commit_choose_card()
            except Exception:
                try:
                    click.cancel_click()
                except Exception:
                    pass
                raise
    except Exception as exc:
        manual_controller.output(f"留牌鼠标操作失败：{exc}")
        return FSM_ERROR
    return wait_until_battle_starts()


def run_manual_battle_step():
    snapshot = refresh_snapshot()
    if snapshot is None:
        return FSM_ERROR
    if snapshot.is_end:
        return FSM_QUITTING_BATTLE
    if not snapshot.is_my_turn:
        return None

    manual_controller.output(snapshot.format_for_manual_control())
    action = manual_controller.prompt_turn_action(snapshot)
    action = manual_controller.bind_to_turn(action, snapshot)
    fresh_snapshot = refresh_snapshot()
    if fresh_snapshot is None:
        return FSM_ERROR
    revision_before = log_state.revision
    result = manual_controller.execute(action, fresh_snapshot)
    manual_controller.output(result.message)
    if result.recovery_needed:
        return FSM_ERROR
    if result.executed:
        wait_for_log_update(revision_before)
    return None


def _report_automation_diagnostic(code, message):
    """Report a stable automation state once instead of every loop."""
    global last_automation_diagnostic
    if last_automation_diagnostic == code:
        return
    manual_controller.output(message)
    last_automation_diagnostic = code


def run_automatic_battle_step():
    """Observe opponent turns; execute one newly validated player action."""
    global player_turn_delay_key, last_automation_diagnostic

    snapshot = refresh_snapshot()
    if snapshot is None:
        _report_automation_diagnostic(
            "power_log_unavailable", "Power.log 暂不可用，继续重试。")
        return None
    if snapshot.is_end:
        return FSM_QUITTING_BATTLE
    if not snapshot.is_my_turn:
        _report_automation_diagnostic("opponent_turn", "等待对手操作。")
        return None
    _report_automation_diagnostic("my_turn", "轮到己方操作：开始读取推荐……")
    turn = snapshot.game_num_turns_in_play
    if player_turn_delay_key != turn:
        # 每个新回合开始只延时一次（给盒子更新推荐留时间），
        # 同回合内的多次出牌操作之间不重复延时。
        player_turn_delay_key = turn
        # 换牌结束后（第一个自己回合）切换小截图模式；换牌期不启用。
        if recommendation_capture.strategy_roi is None:
            recommendation_capture.set_strategy_roi(
                recommendation_config.strategy_roi)
        manual_controller.output(
            f"回合 {turn} 开始：延时 "
            f"{recommendation_config.pre_action_delay_seconds:.0f}s 后开始 OCR……")
        time.sleep(recommendation_config.pre_action_delay_seconds)
    if recommendation_flow is None:
        return run_manual_battle_step()

    result = recommendation_flow.run_player_turn_step()
    if result.status == FlowStepStatus.RETRY:
        if result.diagnostics == "discover_choice_still_open":
            message = "发现选择仍在，准备重新点击。"
        else:
            message = (
                "当前推荐暂不可执行，继续重试："
                f"{result.diagnostics}")
        _report_automation_diagnostic(
            f"retry:{result.diagnostics}", message)
    elif result.status == FlowStepStatus.OBSERVE:
        observe_messages = {
            "opponent_turn": "等待对手操作。",
            "waiting_recommendation_update": "等待盒子更新推荐。",
            "stale_mulligan_recommendation": "等待盒子刷新对局推荐。",
        }
        message = observe_messages.get(
            result.diagnostics,
            f"自动流程观察中：{result.diagnostics}")
        _report_automation_diagnostic(
            f"observe:{result.diagnostics}", message)
    else:
        last_automation_diagnostic = None
    return None


def Battling():
    global win_count

    print_out()
    while True:
        if quitting_flag:
            sys.exit(0)
        next_state = run_automatic_battle_step()
        if next_state == FSM_QUITTING_BATTLE:
            if log_state.my_entity.query_tag("PLAYSTATE") == "WON":
                win_count += 1
                info_print("你赢得了这场对战")
            else:
                info_print("你输了")
            return next_state
        if next_state == FSM_ERROR:
            return next_state
        time.sleep(0.2)


def QuittingBattle():
    print_out()

    time.sleep(5)

    loop_count = 0
    while True:
        if quitting_flag or stop_after_current_game:
            sys.exit(0)

        state = get_screen.get_state()
        if state in [FSM_CHOOSING_HERO, FSM_LEAVE_HS]:
            return state
        click.run_hearthstone_action(lambda: (
            click.cancel_click(),
            click.test_click(),
            click.commit_error_report(),
        ))

        loop_count += 1
        if loop_count >= 15:
            return FSM_ERROR

        time.sleep(STATE_CHECK_INTERVAL+random.random()+random.random()+random.random())


def GoBackHSAction():
    global FSM_state

    print_out()
    time.sleep(3)

    while not get_screen.test_hs_available():
        if quitting_flag or stop_after_current_game:
            sys.exit(0)
        click.enter_HS()
        time.sleep(10)

    # 有时候炉石进程会直接重写Power.log, 这时应该重新创建文件操作句柄
    init()

    return FSM_WAIT_MAIN_MENU


def MainMenuAction():
    print_out()

    time.sleep(3)

    while True:
        if quitting_flag or stop_after_current_game:
            sys.exit(0)

        click.run_hearthstone_action(click.enter_battle_mode)
        time.sleep(5)

        state = get_screen.get_state()

        # 重新连接对战之类的
        if state == FSM_BATTLING:
            ok = update_log_state()
            if ok and log_state.available:
                return FSM_BATTLING
        if state == FSM_CHOOSING_HERO:
            return FSM_CHOOSING_HERO


def WaitMainMenu():
    print_out()
    wait_main_menu_count = 0
    while get_screen.get_state() != FSM_MAIN_MENU:
        click.run_hearthstone_action(click.enter_battle_mode)
        time.sleep(5)
        wait_main_menu_count += 1
        if wait_main_menu_count >= 5:
            break
    return FSM_MAIN_MENU


def HandleErrorAction():
    print_out()

    if not get_screen.test_hs_available():
        return FSM_LEAVE_HS
    manual_controller.output("状态暂不可确认，等待后重新检测。")
    time.sleep(STATE_CHECK_INTERVAL)
    state = get_screen.get_state()
    known_states = {
        FSM_LEAVE_HS, FSM_MAIN_MENU, FSM_CHOOSING_HERO, FSM_MATCHING,
        FSM_CHOOSING_CARD, FSM_BATTLING, FSM_QUITTING_BATTLE,
        FSM_WAIT_MAIN_MENU,
    }
    return state if state in known_states else FSM_ERROR


def FSM_dispatch(next_state):
    dispatch_dict = {
        FSM_LEAVE_HS: GoBackHSAction,
        FSM_MAIN_MENU: MainMenuAction,
        FSM_CHOOSING_HERO: ChoosingHeroAction,
        FSM_MATCHING: MatchingAction,
        FSM_CHOOSING_CARD: ChoosingCardAction,
        FSM_BATTLING: Battling,
        FSM_ERROR: HandleErrorAction,
        FSM_QUITTING_BATTLE: QuittingBattle,
        FSM_WAIT_MAIN_MENU: WaitMainMenu,
    }

    debug_print(f"当前状态为：+{next_state}")
    if next_state not in dispatch_dict:
        error_print("Unknown state!")
        return FSM_ERROR
    else:
        return dispatch_dict[next_state]()


def AutoHS_automata():
    global FSM_state, quitting_flag

    if get_screen.test_hs_available():
        hs_hwnd = get_screen.get_HS_hwnd()
        get_screen.move_window_foreground(hs_hwnd)
        time.sleep(0.5+random.random())

    # 出现这些状态时对局一定不在进行中，满足“打完本局再停止”的条件
    between_game_states = (
        FSM_MAIN_MENU, FSM_CHOOSING_HERO, FSM_MATCHING,
        FSM_WAIT_MAIN_MENU, FSM_LEAVE_HS, "",
    )

    while 1:
        if quitting_flag:
            sys.exit(0)
        if stop_after_current_game and FSM_state in between_game_states:
            info_print("已到计划停止时间，本局对战已经结束，自动化停止。")
            quitting_flag = True
            shutdown_event.set()
            sys.exit(0)
        if FSM_state == "":
            FSM_state = get_screen.get_state()
        FSM_state = FSM_dispatch(FSM_state)





if __name__ == "__main__":
    keyboard.add_hotkey("ctrl+q", system_exit)

    init()
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class KpiConfig:
    horizon_days: int
    target_return: float
    price_source: str  # "high" or "close"
    trigger_mode: str  # "touch" or "end"
    fail_action: str   # "watchlist" or "exit"
    grace_days: int
    require_profit_days: int


def _ret_from_price(entry_px: float, px: float, side: str) -> float:
    if entry_px <= 0:
        return 0.0
    if side == "long":
        return (float(px) - float(entry_px)) / float(entry_px)
    return (float(entry_px) - float(px)) / float(entry_px)


def init_kpi_state() -> Dict[str, object]:
    return {
        "days_elapsed": 0,
        "profit_days": 0,
        "progress": 0.0,
        "passed": False,
        "failed": False,
        "reason": None,
    }


def update_kpi_state(
    *,
    state: Dict[str, object],
    cfg: KpiConfig,
    side: str,
    entry_px: float,
    price_high: Optional[float],
    price_close: Optional[float],
) -> Dict[str, object]:
    if state.get("passed") or state.get("failed"):
        return state

    price = price_high if cfg.price_source == "high" else price_close
    if price is None:
        return state

    # 每個交易日推進一天
    state["days_elapsed"] = int(state.get("days_elapsed", 0)) + 1

    # 進度：取最佳報酬
    ret_now = _ret_from_price(entry_px, float(price), side)
    state["progress"] = max(float(state.get("progress", 0.0)), float(ret_now))

    # profit_days：用 close 來計算正報酬天數
    if price_close is not None:
        ret_close = _ret_from_price(entry_px, float(price_close), side)
        if ret_close > 0:
            state["profit_days"] = int(state.get("profit_days", 0)) + 1

    # 觸發模式：touch
    if cfg.trigger_mode == "touch":
        if ret_now >= cfg.target_return and int(state.get("profit_days", 0)) >= cfg.require_profit_days:
            state["passed"] = True
            state["reason"] = "kpi_touch_pass"
            return state
        # 超過期限仍未達標 -> fail
        if int(state["days_elapsed"]) > int(cfg.horizon_days) + int(cfg.grace_days):
            state["failed"] = True
            state["reason"] = "kpi_timeout"
        return state

    # 觸發模式：end
    if int(state["days_elapsed"]) >= int(cfg.horizon_days):
        if ret_now >= cfg.target_return and int(state.get("profit_days", 0)) >= cfg.require_profit_days:
            state["passed"] = True
            state["reason"] = "kpi_end_pass"
        else:
            if int(state["days_elapsed"]) >= int(cfg.horizon_days) + int(cfg.grace_days):
                state["failed"] = True
                state["reason"] = "kpi_end_fail"

    return state

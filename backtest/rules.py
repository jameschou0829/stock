from __future__ import annotations

from typing import Dict

from typing import Any


def has_any_strategy(row: dict, side: str) -> bool:
    if side == "long":
        keys = [
            "strat_volume_momentum",
            "strat_price_volume_new_high",
            "strat_breakout_edge",
            "strat_trust_breakout",
            "strat_trust_momentum_buy",
            "strat_foreign_big_buy",
            "strat_co_buy",
        ]
    else:
        keys = [
            "strat_volume_momentum_weak",
            "strat_price_volume_new_low",
            "strat_trust_breakdown",
            "strat_trust_momentum_sell",
            "strat_foreign_big_sell",
            "strat_co_sell",
        ]
    return any(int(row.get(k, 0) or 0) == 1 for k in keys)


def has_momentum(row: dict, side: str) -> bool:
    """
    「沒量沒動能」的判斷：當天沒有任何策略旗標，且沒有量突破/價突破。
    """
    if has_any_strategy(row, side):
        return True
    if int(row.get("is_volume_breakout_20d", 0) or 0) == 1:
        return True
    if side == "long" and int(row.get("is_price_breakout_20d", 0) or 0) == 1:
        return True
    if side == "short" and int(row.get("is_price_breakdown_20d", 0) or 0) == 1:
        return True
    return False


def evaluate_entry(row: dict, side: str, cfg: Any) -> bool:
    regime = row.get("market_regime") or "unknown"
    score_long = int(row.get("score_long", row.get("score", 0)) or 0)
    score_short = int(row.get("score_short", abs(int(row.get("score", 0) or 0))) or 0)
    min_long = int(cfg.entry_min_score_long or cfg.entry_min_score or 0)
    min_short = int(cfg.entry_min_score_short or cfg.entry_min_score or 0)
    if side == "long":
        if regime not in cfg.allowed_regimes_long:
            return False
        if score_long < min_long:
            return False
    else:
        if regime not in cfg.allowed_regimes_short:
            return False
        if score_short < min_short:
            return False
    return has_any_strategy(row, side)

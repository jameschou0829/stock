from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional, Tuple

from utils.db import db_conn
from utils.db_schema import has_column
from backtest.rules import evaluate_entry


def _score_fields(side: str, has_score_long: bool, has_score_short: bool) -> Tuple[str, str]:
    score_long_expr = "score_long" if has_score_long else "score"
    score_short_expr = "score_short" if has_score_short else "ABS(score)"
    score_expr = score_long_expr if side == "long" else score_short_expr
    return score_expr, score_long_expr


def select_candidates(
    *,
    trading_date: date,
    side: str,
    cfg,
    limit: Optional[int] = None,
    signals_by_date: Optional[Dict[date, Dict[str, dict]]] = None,
) -> List[dict]:
    """
    統一選股邏輯：
    - entry_flag 先過濾（entry_long/entry_short）
    - evaluate_entry 做門檻/市場條件
    - score_long/score_short 排序
    """
    entry_col = "entry_long" if side == "long" else "entry_short"
    limit = int(limit or cfg.max_positions or cfg.top_n)

    if signals_by_date is not None:
        rows = list(signals_by_date.get(trading_date, {}).values())
    else:
        has_score_long = has_column("stock_signals_v2", "score_long")
        has_score_short = has_column("stock_signals_v2", "score_short")
        has_rationale_tags = has_column("stock_signals_v2", "rationale_tags")
        has_prob = has_column("stock_signals_v2", cfg.prob_field) if getattr(cfg, "prob_field", "") else False
        score_expr, score_long_expr = _score_fields(side, has_score_long, has_score_short)
        score_short_expr = "score_short" if has_score_short else "ABS(score)"
        rationale_col = "rationale_tags" if has_rationale_tags else "NULL AS rationale_tags"
        prob_col = (f"{cfg.prob_field} AS entry_prob") if has_prob else "NULL AS entry_prob"
        with db_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT stock_id, trading_date, market_regime,
                           {score_long_expr} AS score_long,
                           {score_short_expr} AS score_short,
                           {score_expr} AS score_value,
                           entry_long, entry_short,
                           primary_strategy,
                           {rationale_col},
                           {prob_col},
                           strat_volume_momentum, strat_price_volume_new_high, strat_breakout_edge, strat_trust_breakout, strat_trust_momentum_buy,
                           strat_foreign_big_buy, strat_co_buy,
                           strat_volume_momentum_weak, strat_price_volume_new_low, strat_trust_breakdown, strat_trust_momentum_sell,
                           strat_foreign_big_sell, strat_co_sell
                    FROM stock_signals_v2
                    WHERE trading_date=%s
                    """,
                    (trading_date,),
                )
                rows = cur.fetchall() or []

    candidates: List[dict] = []
    for row in rows:
        strategy = getattr(cfg, "strategy", None)
        if strategy:
            if int(row.get(strategy, 0) or 0) != 1:
                continue
        if int(row.get(entry_col, 0) or 0) != 1:
            continue
        if not evaluate_entry(row, side, cfg):
            continue
        score_val = float(row.get("score_value", row.get("score_long" if side == "long" else "score_short", 0)) or 0.0)
        if score_val < float(cfg.min_abs_score):
            continue
        prob_gte = getattr(cfg, "prob_gte", None)
        if prob_gte is not None:
            p = row.get("entry_prob")
            if p is None:
                continue
            try:
                if float(p) < float(prob_gte):
                    continue
            except Exception:
                continue
        candidates.append(row)

    candidates.sort(key=lambda r: float(r.get("score_value", r.get("score_long" if side == "long" else "score_short", 0)) or 0.0), reverse=True)
    return candidates[:limit]

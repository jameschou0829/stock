from __future__ import annotations

import os
import sys
import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import settings
from configs.strategy_loader import load_strategy_config
from utils.db import db_conn
from backtest.kpi import KpiConfig, init_kpi_state, update_kpi_state
from backtest.rules import has_momentum, evaluate_entry
from backtest.selection import select_candidates
from utils.db_schema import has_column


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fmt_date(d: date) -> str:
    return d.strftime("%Y-%m-%d")


@dataclass
class Trade:
    side: str  # "long" or "short"
    stock_id: str
    signal_date: date
    entry_date: date
    exit_date: date
    entry_px: float
    exit_px: float
    ret: float
    stopped: bool


@dataclass
class BacktestConfig:
    start: date
    end: date
    side: str = "long"  # long/short/both
    top_n: int = 20
    holding_days: int = 0
    min_abs_score: int = 0
    entry_min_score: int = 0
    entry_min_score_long: int = 0
    entry_min_score_short: int = 0
    use_stop_loss: bool = False
    use_entry_exit_signals: bool = True
    exit_no_momentum_days: int = 2
    calendar_stock_id: str = getattr(settings, "MARKET_PROXY_STOCK_ID", "0050")
    allowed_regimes_long: Tuple[str, ...] = ("bull", "range")
    allowed_regimes_short: Tuple[str, ...] = ("bear", "range")

    # ===== Phase A：可信回測參數 =====
    initial_capital: float = 0.0
    tranches: int = 0
    max_positions: int = 0
    stop_loss_pct: float = 0.0
    equity_stop_pct: float = 0.0  # 累積虧損達 -30% 停止開新倉（仍管理既有倉）
    commission_bps: float = 0.0   # 0.1425%
    tax_bps: float = 0.0          # 0.30%（僅賣出收稅，short 進場視為賣出）
    slippage_bps: float = 0.0     # 0.05%

    # KPI / rotation
    kpi: KpiConfig | None = None
    rotation_allow_daily: bool = False
    rotation_delta_edge: float = 0.0
    rotation_requires_fail: bool = True
    rotation_edge_source: str = "score"  # score / p_kpi
    prob_field: str = ""
    entry_timing: str = "next_open"
    persist_run: bool = True
    strategy: Optional[str] = None
    prob_gte: Optional[float] = None

    # config snapshot
    config_hash: str = ""
    config_snapshot: str = ""


def build_backtest_config(
    *,
    start: date,
    end: date,
    side: str = "long",
    overrides: Optional[dict] = None,
    top_n: Optional[int] = None,
    holding_days: Optional[int] = None,
    min_abs_score: Optional[int] = None,
    entry_min_score: Optional[int] = None,
    exit_no_momentum_days: Optional[int] = None,
    use_stop_loss: Optional[bool] = None,
    use_entry_exit_signals: Optional[bool] = None,
    calendar_stock_id: Optional[str] = None,
    strategy: Optional[str] = None,
    prob_gte: Optional[float] = None,
) -> BacktestConfig:
    strategy_cfg, config_hash, config_snapshot = load_strategy_config(overrides)

    costs = strategy_cfg.get("costs", {})
    portfolio = strategy_cfg.get("portfolio", {})
    risk = strategy_cfg.get("risk", {})
    scoring = strategy_cfg.get("scoring", {})
    kpi_raw = strategy_cfg.get("kpi", {})
    rotation_raw = strategy_cfg.get("rotation", {})
    backtest_raw = strategy_cfg.get("backtest", {})
    prob_raw = strategy_cfg.get("probability", {})
    cfg_prob_gte = prob_raw.get("min_prob_gte", None)

    entry_long = int(scoring.get("entry_min_score_long", 0) or 0)
    entry_short = int(scoring.get("entry_min_score_short", 0) or 0)

    kpi_cfg = KpiConfig(
        horizon_days=int(kpi_raw.get("horizon_days", 0) or 0),
        target_return=float(kpi_raw.get("target_return", 0.0) or 0.0),
        price_source=str(kpi_raw.get("price_source", "close")),
        trigger_mode=str(kpi_raw.get("trigger_mode", "touch")),
        fail_action=str(kpi_raw.get("fail_action", "watchlist")),
        grace_days=int(kpi_raw.get("grace_days", 0) or 0),
        require_profit_days=int(kpi_raw.get("require_profit_days", 0) or 0),
    )

    cfg = BacktestConfig(
        start=start,
        end=end,
        side=side,
        top_n=int(top_n) if top_n is not None else int(portfolio.get("max_positions", 0) or 0),
        holding_days=int(holding_days) if holding_days is not None else int(backtest_raw.get("holding_days", 0) or 0),
        min_abs_score=int(min_abs_score) if min_abs_score is not None else int(backtest_raw.get("min_abs_score", 0) or 0),
        entry_min_score=int(entry_min_score) if entry_min_score is not None else entry_long,
        entry_min_score_long=entry_long,
        entry_min_score_short=entry_short,
        use_stop_loss=bool(use_stop_loss) if use_stop_loss is not None else bool(backtest_raw.get("use_stop_loss", True)),
        use_entry_exit_signals=bool(use_entry_exit_signals) if use_entry_exit_signals is not None else bool(backtest_raw.get("use_entry_exit_signals", True)),
        exit_no_momentum_days=int(exit_no_momentum_days) if exit_no_momentum_days is not None else int(backtest_raw.get("exit_no_momentum_days", 2) or 2),
        calendar_stock_id=str(calendar_stock_id) if calendar_stock_id else str(strategy_cfg.get("signals", {}).get("market_proxy_stock_id", getattr(settings, "MARKET_PROXY_STOCK_ID", "0050"))),
        allowed_regimes_long=tuple(scoring.get("allowed_regimes_long", ["bull", "range"])),
        allowed_regimes_short=tuple(scoring.get("allowed_regimes_short", ["bear", "range"])),
        initial_capital=float(portfolio.get("initial_capital", 0.0) or 0.0),
        tranches=int(portfolio.get("tranches", 0) or 0),
        max_positions=int(portfolio.get("max_positions", 0) or 0),
        stop_loss_pct=float(risk.get("stop_loss_pct", 0.0) or 0.0),
        equity_stop_pct=float(portfolio.get("equity_stop_pct", 0.0) or 0.0),
        commission_bps=float(costs.get("commission_bps", 0.0) or 0.0),
        tax_bps=float(costs.get("tax_bps", 0.0) or 0.0),
        slippage_bps=float(costs.get("slippage_bps", 0.0) or 0.0),
        kpi=kpi_cfg,
        rotation_allow_daily=bool(rotation_raw.get("allow_daily_rotation", False)),
        rotation_delta_edge=float(rotation_raw.get("delta_edge", 0.0) or 0.0),
        rotation_requires_fail=bool(rotation_raw.get("rotation_requires_fail", True)),
        rotation_edge_source=str(rotation_raw.get("edge_source", "score")),
        prob_field=str(prob_raw.get("prob_field", "")),
        entry_timing=str(backtest_raw.get("entry_timing", "next_open")),
        strategy=strategy,
        prob_gte=float(prob_gte) if prob_gte is not None else (float(cfg_prob_gte) if cfg_prob_gte is not None else None),
        config_hash=config_hash,
        config_snapshot=config_snapshot,
    )
    return cfg


def _apply_cost(amount: float, bps: float) -> float:
    return float(amount) * float(bps) / 10000.0


def _trade_costs(notional: float, *, side: str, action: str, commission_bps: float, tax_bps: float, slippage_bps: float) -> float:
    """
    簡化版成本模型（以 notional 計）：
    - commission：進出都收
    - slippage：進出都算（當作成交價不利）
    - tax：台股現股賣出收稅；short 進場等同賣出，也收稅；short 出場買回不收
    """
    c = _apply_cost(notional, commission_bps) + _apply_cost(notional, slippage_bps)
    if action == "exit" and side == "long":
        c += _apply_cost(notional, tax_bps)
    if action == "entry" and side == "short":
        c += _apply_cost(notional, tax_bps)
    return float(c)


def _stop_price(entry_px: float, *, side: str, stop_loss_pct: float) -> float:
    if side == "long":
        return float(entry_px) * (1.0 - float(stop_loss_pct))
    return float(entry_px) * (1.0 + float(stop_loss_pct))


def _hit_stop(bar: dict, *, side: str, stop_px: float) -> Optional[float]:
    """
    回傳停損成交價（考慮跳空），未觸發回 None。
    """
    lo = pick_price(bar, "low")
    hi = pick_price(bar, "high")
    op = pick_price(bar, "open")
    if lo is None or hi is None:
        return None
    if side == "long" and lo <= stop_px:
        if op is not None and op <= stop_px:
            return float(op)
        return float(stop_px)
    if side == "short" and hi >= stop_px:
        if op is not None and op >= stop_px:
            return float(op)
        return float(stop_px)
    return None


def get_trading_calendar(start: date, end: date, calendar_stock_id: str) -> List[date]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT trading_date
                FROM stock_daily
                WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (calendar_stock_id, start, end),
            )
            rows = cur.fetchall()
            return [r["trading_date"] for r in rows]


def next_calendar_date(cal: List[date], d: date, offset: int) -> Optional[date]:
    try:
        idx = cal.index(d)
    except ValueError:
        return None
    j = idx + offset
    if j < 0 or j >= len(cal):
        return None
    return cal[j]


def pick_price(bar: dict, which: str) -> Optional[float]:
    v = bar.get(which)
    return float(v) if v is not None else None


def _has_column(conn, table_name: str, column_name: str) -> bool:
    return has_column(table_name, column_name)


def rotation_should_trigger(
    *,
    best_edge: float,
    worst_edge: float,
    worst_failed: bool,
    worst_momentum: bool,
    delta_edge: float,
    requires_fail: bool,
) -> bool:
    if (best_edge - worst_edge) < float(delta_edge):
        return False
    if requires_fail and (not worst_failed and worst_momentum):
        return False
    return True


def evaluate_rotation(
    *,
    best_edge: float,
    worst_edge: float,
    worst_failed: bool,
    worst_momentum: bool,
    delta_edge: float,
    requires_fail: bool,
) -> tuple[bool, Optional[str]]:
    if rotation_should_trigger(
        best_edge=best_edge,
        worst_edge=worst_edge,
        worst_failed=worst_failed,
        worst_momentum=worst_momentum,
        delta_edge=delta_edge,
        requires_fail=requires_fail,
    ):
        return True, "rotation"
    return False, None


def get_signals_candidates_for_date(trading_date: date, side: str, top_n: int, min_abs_score: int) -> List[dict]:
    cfg = BacktestConfig(start=trading_date, end=trading_date, side=side)
    cfg.top_n = top_n
    cfg.min_abs_score = min_abs_score
    return select_candidates(trading_date=trading_date, side=side, cfg=cfg, limit=top_n)




def get_daily_bars(stock_id: str, start: date, end: date) -> List[dict]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, open, high, low, close
                FROM stock_daily
                WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (stock_id, start, end),
            )
            return cur.fetchall()


def get_signals_rows(stock_id: str, start: date, end: date) -> Dict[date, dict]:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, exit_long, exit_short,
                       stop_loss_price,
                       is_volume_breakout_20d, is_price_breakout_20d, is_price_breakdown_20d,
                       strat_volume_momentum, strat_price_volume_new_high, strat_breakout_edge, strat_trust_breakout, strat_trust_momentum_buy,
                       strat_foreign_big_buy, strat_co_buy,
                       strat_volume_momentum_weak, strat_price_volume_new_low, strat_trust_breakdown, strat_trust_momentum_sell,
                       strat_foreign_big_sell, strat_co_sell
                FROM stock_signals_v2
                WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (stock_id, start, end),
            )
            rows = cur.fetchall()
            return {r["trading_date"]: r for r in rows}


def simulate_exit(
    side: str,
    stock_id: str,
    entry_date: date,
    max_exit_date: date,
    stop_loss_price: Optional[float],
    use_stop_loss: bool,
) -> Optional[Tuple[date, float, bool, float]]:
    """
    回傳 (exit_date, exit_px, stopped, entry_px)；出場價用 close（或停損價）。
    """
    bars = get_daily_bars(stock_id, entry_date, max_exit_date)
    if not bars:
        return None

    by_date = {b["trading_date"]: b for b in bars}
    eb = by_date.get(entry_date)
    if not eb:
        return None
    entry_px = pick_price(eb, "open")
    if entry_px is None or entry_px <= 0:
        return None

    sig_map = get_signals_rows(stock_id, entry_date, max_exit_date)

    for b in bars:
        d = b["trading_date"]
        lo = pick_price(b, "low")
        hi = pick_price(b, "high")
        cl = pick_price(b, "close")
        op = pick_price(b, "open")
        if cl is None:
            continue

        if use_stop_loss and stop_loss_price is not None and lo is not None and hi is not None:
            sl = float(stop_loss_price)
            if side == "long" and lo <= sl:
                # 跳空風險：若開盤就已經跌破停損，實務上會以開盤或更差成交
                if op is not None and op <= sl:
                    return d, float(op), True, float(entry_px)
                return d, sl, True, float(entry_px)
            if side == "short" and hi >= sl:
                # 跳空風險：若開盤就已經站上停損，實務上會以開盤或更差成交
                if op is not None and op >= sl:
                    return d, float(op), True, float(entry_px)
                return d, sl, True, float(entry_px)

        sig = sig_map.get(d)
        if sig:
            if side == "long" and int(sig.get("exit_long", 0) or 0) == 1:
                return d, float(cl), False, float(entry_px)
            if side == "short" and int(sig.get("exit_short", 0) or 0) == 1:
                return d, float(cl), False, float(entry_px)

    xb = by_date.get(max_exit_date)
    if not xb:
        return None
    exit_px = pick_price(xb, "close")
    if exit_px is None or exit_px <= 0:
        return None
    return max_exit_date, float(exit_px), False, float(entry_px)


def get_signals_for_stocks_on_date(stock_ids: List[str], trading_date: date) -> Dict[str, dict]:
    if not stock_ids:
        return {}
    with db_conn() as conn:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(stock_ids))
            cur.execute(
                f"""
                SELECT *
                FROM stock_signals_v2
                WHERE trading_date=%s AND stock_id IN ({placeholders})
                """,
                (trading_date, *stock_ids),
            )
            rows = cur.fetchall()
            return {r["stock_id"]: r for r in rows}


def get_daily_close_for_stocks_on_date(stock_ids: List[str], trading_date: date) -> Dict[str, dict]:
    if not stock_ids:
        return {}
    with db_conn() as conn:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(stock_ids))
            cur.execute(
                f"""
                SELECT stock_id, trading_date, open, high, low, close
                FROM stock_daily
                WHERE trading_date=%s AND stock_id IN ({placeholders})
                """,
                (trading_date, *stock_ids),
            )
            rows = cur.fetchall()
            return {r["stock_id"]: r for r in rows}


def run_backtest_daily(cfg: BacktestConfig) -> dict:
    """
    holding_days=0：不固定持有天數
    - 每日根據 signals 產生 entry（隔日開盤進）
    - 每日根據 exit 或「沒量沒動能連續 N 天」或停損出場
    - 等權投組日淨值（以 close 做 mark-to-market）
    """
    cal = get_trading_calendar(cfg.start, cfg.end, cfg.calendar_stock_id)
    if len(cal) < 3:
        raise ValueError("交易日太少")
    date_to_idx = {d: i for i, d in enumerate(cal)}

    # ===== Prefetch（禁止迴圈內 N+1 DB）=====
    start0 = cfg.start
    end0 = cfg.end
    end_plus = cal[min(len(cal) - 1, date_to_idx.get(end0, len(cal) - 1) + 1)]

    with db_conn() as conn:
        has_score_long = _has_column(conn, "stock_signals_v2", "score_long")
        has_score_short = _has_column(conn, "stock_signals_v2", "score_short")
        has_rationale_tags = _has_column(conn, "stock_signals_v2", "rationale_tags")
        has_prob = bool(cfg.prob_field) and _has_column(conn, "stock_signals_v2", cfg.prob_field)
        score_long_expr = "score_long" if has_score_long else "score"
        score_short_expr = "score_short" if has_score_short else "ABS(score)"
        rationale_col = "rationale_tags" if has_rationale_tags else "NULL AS rationale_tags"
        prob_col = (f"{cfg.prob_field} AS entry_prob") if has_prob else "NULL AS entry_prob"
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stock_id, trading_date, open, high, low, close
                FROM stock_daily
                WHERE trading_date BETWEEN %s AND %s
                """,
                (start0, end_plus),
            )
            bar_rows = cur.fetchall() or []
            cur.execute(
                f"""
                SELECT stock_id, trading_date, market_regime,
                       {score_long_expr} AS score_long,
                       {score_short_expr} AS score_short,
                       entry_long, entry_short, exit_long, exit_short,
                       primary_strategy, {rationale_col}, {prob_col},
                       is_volume_breakout_20d, is_price_breakout_20d, is_price_breakdown_20d,
                       strat_volume_momentum, strat_price_volume_new_high, strat_breakout_edge, strat_trust_breakout, strat_trust_momentum_buy,
                       strat_foreign_big_buy, strat_co_buy,
                       strat_volume_momentum_weak, strat_price_volume_new_low, strat_trust_breakdown, strat_trust_momentum_sell,
                       strat_foreign_big_sell, strat_co_sell
                FROM stock_signals_v2
                WHERE trading_date BETWEEN %s AND %s
                """,
                (start0, end0),
            )
            sig_rows = cur.fetchall() or []

    bars_by_date: Dict[date, Dict[str, dict]] = {}
    for r in bar_rows:
        bars_by_date.setdefault(r["trading_date"], {})[r["stock_id"]] = r
    sig_by_date: Dict[date, Dict[str, dict]] = {}
    for r in sig_rows:
        sig_by_date.setdefault(r["trading_date"], {})[r["stock_id"]] = r

    # ===== 資金/狀態機 =====
    initial = float(cfg.initial_capital)
    cash = float(cfg.initial_capital)
    equity = float(cfg.initial_capital)
    tranche_value = initial / float(cfg.tranches) if cfg.tranches > 0 else 0.0
    halted = False

    positions: Dict[str, dict] = {}  # stock_id -> pos
    no_mom_cnt: Dict[str, int] = {}
    trades: List[dict] = []
    equity_curve = []
    peak_equity = equity
    total_cost = 0.0
    total_turnover = 0.0

    pending_entries: set[str] = set()
    pending_entry_signal_date: Optional[date] = None

    def can_open_new_positions() -> bool:
        nonlocal halted
        if halted:
            return False
        if equity <= initial * (1.0 - float(cfg.equity_stop_pct)):
            halted = True
            return False
        return True

    def _close_position(*, sid: str, pos: dict, exit_px: float, exit_date: date, reason: str, stopped: bool = False):
        nonlocal cash, total_cost, total_turnover
        side = pos["side"]
        notional = float(pos["notional"])
        entry_px = float(pos["entry_px"])
        entry_cost = float(pos.get("entry_cost", 0.0))
        ret_gross = (float(exit_px) - entry_px) / entry_px if side == "long" else (entry_px - float(exit_px)) / entry_px
        exit_notional = notional * (1.0 + ret_gross)
        exit_cost = _trade_costs(
            exit_notional,
            side=side,
            action="exit",
            commission_bps=cfg.commission_bps,
            tax_bps=cfg.tax_bps,
            slippage_bps=cfg.slippage_bps,
        )
        total_cost += float(entry_cost) + float(exit_cost)
        total_turnover += float(notional) + float(exit_notional)
        cash += exit_notional - exit_cost
        ret_net = ret_gross - ((entry_cost + exit_cost) / notional if notional > 0 else 0.0)
        trades.append(
            {
                "side": side,
                "stock_id": sid,
                "signal_date": fmt_date(pos.get("signal_date")),
                "entry_exec_date": fmt_date(pos.get("entry_exec_date")),
                "entry_timing": pos.get("entry_timing"),
                "entry_primary_strategy": pos.get("entry_primary_strategy"),
                "entry_rationale_tags": pos.get("entry_rationale_tags"),
                "entry_score": pos.get("entry_score"),
                "entry_prob": pos.get("entry_prob"),
                "exit_date": fmt_date(exit_date),
                "entry_price": float(entry_px),
                "exit_price": float(exit_px),
                "ret_gross": float(ret_gross),
                "ret_net": float(ret_net),
                "cost_paid": float(entry_cost + exit_cost),
                "exit_reason": reason,
                "kpi_passed": bool(pos.get("kpi_passed")),
                "kpi_failed": bool(pos.get("kpi_failed")),
                "kpi_state": pos.get("kpi_state"),
                "stopped": bool(stopped),
            }
        )

    def _open_position(*, sid: str, side: str, exec_date: date, entry_px: float, sig_row: dict):
        nonlocal cash, total_cost
        notional = float(tranche_value)
        if notional <= 0 or cash < notional:
            return False
        stop_px = _stop_price(float(entry_px), side=side, stop_loss_pct=float(cfg.stop_loss_pct))
        entry_cost = _trade_costs(
            notional,
            side=side,
            action="entry",
            commission_bps=cfg.commission_bps,
            tax_bps=cfg.tax_bps,
            slippage_bps=cfg.slippage_bps,
        )
        cash -= notional
        cash -= entry_cost
        total_cost += float(entry_cost)
        entry_score = float(sig_row.get("score_long" if side == "long" else "score_short", 0) or 0.0)
        positions[sid] = {
            "side": side,
            "signal_date": sig_row.get("trading_date"),
            "entry_exec_date": exec_date,
            "entry_timing": cfg.entry_timing,
            "entry_px": float(entry_px),
            "last_close": float(sig_row.get("close") or entry_px),
            "notional": float(notional),
            "stop_px": float(stop_px),
            "primary_strategy": sig_row.get("primary_strategy"),
            "entry_primary_strategy": sig_row.get("primary_strategy"),
            "entry_rationale_tags": sig_row.get("rationale_tags"),
            "entry_score": float(entry_score),
            "entry_prob": sig_row.get("entry_prob"),
            "entry_cost": float(entry_cost),
            "edge": float(entry_score),
            "kpi_state": init_kpi_state(),
            "kpi_passed": False,
            "kpi_failed": False,
            "kpi_reason": None,
        }
        no_mom_cnt[sid] = 0
        return True

    for idx in range(len(cal) - 1):
        d = cal[idx]
        d_next = cal[idx + 1]

        # 0) equity_stop 檢查（只影響開新倉）
        can_open_new_positions()

        # 1) 執行昨日產生的進場（在 d 的開盤進場）
        if cfg.entry_timing == "next_open" and pending_entry_signal_date == d and pending_entries and can_open_new_positions():
            bmap = bars_by_date.get(d, {})
            for sid in list(pending_entries):
                if sid in positions:
                    continue
                if len(positions) >= int(cfg.max_positions):
                    break
                b = bmap.get(sid)
                if not b:
                    continue
                entry_px = pick_price(b, "open")
                if entry_px is None or entry_px <= 0:
                    continue
                sig0 = sig_by_date.get(cal[idx - 1], {}).get(sid) if idx > 0 else None
                side = None
                if sig0:
                    if int(sig0.get("entry_long", 0) or 0) == 1:
                        side = "long"
                    elif int(sig0.get("entry_short", 0) or 0) == 1:
                        side = "short"
                if side is None or sig0 is None:
                    continue
                _open_position(sid=sid, side=side, exec_date=d, entry_px=float(entry_px), sig_row=sig0)
            pending_entries = set()

        # 2) 當天收盤：停損/出場 + KPI
        exits: set[str] = set()
        bmap_today = bars_by_date.get(d, {})
        sig_today_map = sig_by_date.get(d, {})

        for sid, pos in list(positions.items()):
            b = bmap_today.get(sid)
            if not b:
                continue
            cl = pick_price(b, "close")
            if cl is None or cl <= 0:
                continue

            side = pos["side"]

            # 日內停損
            stop_px = float(pos["stop_px"])
            hit_px = _hit_stop(b, side=side, stop_px=stop_px) if cfg.use_stop_loss else None
            if hit_px is not None and pos.get("entry_date") != d:
                _close_position(sid=sid, pos=pos, exit_px=float(hit_px), exit_date=d, reason="stop", stopped=True)
                exits.add(sid)
                continue

            # 避免進場當天被 exit/no_momentum 洗出去（允許停損）
            if pos.get("entry_date") == d:
                pos["last_close"] = float(cl)
                continue

            sig = sig_today_map.get(sid)
            exit_flag = False
            exit_reason = None
            if sig:
                exit_flag = int(sig.get("exit_long" if side == "long" else "exit_short", 0) or 0) == 1
                if exit_flag:
                    exit_reason = "signal_exit"

            # KPI
            if cfg.kpi:
                state = update_kpi_state(
                    state=pos.get("kpi_state", {}),
                    cfg=cfg.kpi,
                    side=side,
                    entry_px=float(pos["entry_px"]),
                    price_high=pick_price(b, "high"),
                    price_close=pick_price(b, "close"),
                )
                pos["kpi_state"] = state
                if state.get("passed"):
                    pos["kpi_passed"] = True
                    pos["kpi_reason"] = state.get("reason")
                if state.get("failed"):
                    pos["kpi_failed"] = True
                    pos["kpi_reason"] = state.get("reason")
                    if cfg.kpi.fail_action == "exit":
                        exit_flag = True
                        exit_reason = "kpi_fail"

            # 動能消失
            if sig and (not exit_flag):
                mom = has_momentum(sig, side)
                if not mom:
                    no_mom_cnt[sid] = int(no_mom_cnt.get(sid, 0)) + 1
                else:
                    no_mom_cnt[sid] = 0
                if int(no_mom_cnt[sid]) >= int(cfg.exit_no_momentum_days):
                    exit_flag = True
                    exit_reason = "time_exit"

            pos["last_close"] = float(cl)

            if exit_flag:
                _close_position(sid=sid, pos=pos, exit_px=float(cl), exit_date=d, reason=exit_reason or "exit")
                exits.add(sid)

        # 2.1 移除出場
        for sid in exits:
            positions.pop(sid, None)
            no_mom_cnt.pop(sid, None)

        # 2.2 Rotation（每日換車）
        if cfg.rotation_allow_daily and can_open_new_positions():
            sig_map = sig_today_map
            for side in ("long", "short"):
                if cfg.side != "both" and cfg.side != side:
                    continue
                pos_side = {sid: p for sid, p in positions.items() if p.get("side") == side}
                if not pos_side:
                    continue
                # worst position
                def _edge_from_sig(sid: str) -> float:
                    s = sig_map.get(sid)
                    if s:
                        return float(s.get("score_long" if side == "long" else "score_short", pos_side[sid].get("edge", 0.0)) or 0.0)
                    return float(pos_side[sid].get("edge", 0.0) or 0.0)

                worst_sid = min(pos_side.keys(), key=_edge_from_sig)
                worst_edge = _edge_from_sig(worst_sid)
                worst_sig = sig_map.get(worst_sid)
                worst_momentum = has_momentum(worst_sig, side) if worst_sig else False
                worst_failed = bool(pos_side[worst_sid].get("kpi_failed"))


                # best candidate（與排行榜一致）
                candidates = select_candidates(trading_date=d, side=side, cfg=cfg, limit=cfg.max_positions, signals_by_date=sig_by_date)
                if not candidates:
                    continue
                best = candidates[0]
                best_sid = best.get("stock_id")
                best_edge = float(best.get("score_long" if side == "long" else "score_short", 0) or 0.0)
                should_rotate, rotation_reason = evaluate_rotation(
                    best_edge=best_edge,
                    worst_edge=worst_edge,
                    worst_failed=worst_failed,
                    worst_momentum=worst_momentum,
                    delta_edge=cfg.rotation_delta_edge,
                    requires_fail=cfg.rotation_requires_fail,
                )
                if not should_rotate:
                    continue

                # 旋轉：先出再進
                b = bmap_today.get(worst_sid)
                if b and b.get("close") is not None:
                    _close_position(sid=worst_sid, pos=pos_side[worst_sid], exit_px=float(b["close"]), exit_date=d, reason=rotation_reason or "rotation")
                    positions.pop(worst_sid, None)
                    no_mom_cnt.pop(worst_sid, None)
                    pending_entries.add(best_sid)

        # 2.3 更新 equity（收盤市值）
        equity = float(cash)
        for sid, pos in positions.items():
            b = bmap_today.get(sid)
            if not b or b.get("close") is None:
                continue
            cl = float(b["close"])
            entry_px = float(pos["entry_px"])
            notional = float(pos["notional"])
            ret = (cl - entry_px) / entry_px if pos["side"] == "long" else (entry_px - cl) / entry_px
            equity += notional * (1.0 + ret)
        peak_equity = max(peak_equity, equity)
        drawdown = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        equity_curve.append(
            {
                "date": fmt_date(d),
                "equity": float(equity),
                "drawdown": float(drawdown),
                "cash": float(cash),
                "positions": int(len(positions)),
                "positions_count": int(len(positions)),
            }
        )

        # 3) 產生進場名單（與 /api/rankings 同一套邏輯）
        pending_entries = set(pending_entries)
        slots = max(0, int(cfg.max_positions) - len(positions) - len(pending_entries))
        if slots > 0 and can_open_new_positions():
            cand_rows: List[dict] = []
            if cfg.side in ("long", "both"):
                rows_long = select_candidates(trading_date=d, side="long", cfg=cfg, limit=slots, signals_by_date=sig_by_date)
                for r in rows_long:
                    rr = dict(r)
                    rr["_side"] = "long"
                    rr["_score"] = float(rr.get("score_long", 0) or 0.0)
                    cand_rows.append(rr)
            if cfg.side in ("short", "both"):
                rows_short = select_candidates(trading_date=d, side="short", cfg=cfg, limit=slots, signals_by_date=sig_by_date)
                for r in rows_short:
                    rr = dict(r)
                    rr["_side"] = "short"
                    rr["_score"] = float(rr.get("score_short", 0) or 0.0)
                    cand_rows.append(rr)

            cand_rows.sort(key=lambda r: float(r.get("_score", 0) or 0.0), reverse=True)
            cand_rows = [r for r in cand_rows if r.get("stock_id") not in positions and r.get("stock_id") not in pending_entries][:slots]

            if cfg.entry_timing == "close":
                bmap_today = bars_by_date.get(d, {})
                if pending_entries:
                    for sid in list(pending_entries):
                        if sid in positions:
                            continue
                        sig_row = sig_by_date.get(d, {}).get(sid)
                        b = bmap_today.get(sid) if bmap_today else None
                        if not sig_row or not b:
                            continue
                        entry_px = pick_price(b, "close")
                        if entry_px is None or entry_px <= 0:
                            continue
                        side = "long" if int(sig_row.get("entry_long", 0) or 0) == 1 else "short"
                        _open_position(sid=sid, side=side, exec_date=d, entry_px=float(entry_px), sig_row=sig_row)
                    pending_entries = set()
                for r in cand_rows:
                    sid = r.get("stock_id")
                    if not sid or sid in positions:
                        continue
                    b = bmap_today.get(sid)
                    if not b:
                        continue
                    entry_px = pick_price(b, "close")
                    if entry_px is None or entry_px <= 0:
                        continue
                    _open_position(sid=sid, side=r.get("_side"), exec_date=d, entry_px=float(entry_px), sig_row=r)
            else:
                for r in cand_rows:
                    if r.get("stock_id"):
                        pending_entries.add(r["stock_id"])
                pending_entry_signal_date = d_next

            if cfg.entry_timing == "next_open":
                pending_entry_signal_date = d_next

    # summary
    wins = sum(1 for t in trades if float(t.get("ret_net", 0.0)) > 0)
    win_rate = (wins / len(trades)) if trades else 0.0
    avg_trade = (sum(float(t.get("ret_net", 0.0)) for t in trades) / len(trades)) if trades else 0.0
    eq = [p["equity"] for p in equity_curve] if equity_curve else [float(cfg.initial_capital)]
    mdd = max_drawdown(eq)
    days = (cfg.end - cfg.start).days if cfg.end and cfg.start else 0
    years = days / 365.0 if days > 0 else 0.0
    cagr = (eq[-1] / float(cfg.initial_capital)) ** (1.0 / years) - 1.0 if years > 0 and cfg.initial_capital > 0 else 0.0
    turnover = (total_turnover / float(cfg.initial_capital)) if cfg.initial_capital > 0 else 0.0
    return {
        "config": {
            "start": fmt_date(cfg.start),
            "end": fmt_date(cfg.end),
            "side": cfg.side,
            "top_n": cfg.top_n,
            "holding_days": cfg.holding_days,
            "min_abs_score": cfg.min_abs_score,
            "entry_min_score": cfg.entry_min_score,
            "entry_min_score_long": cfg.entry_min_score_long,
            "entry_min_score_short": cfg.entry_min_score_short,
            "use_stop_loss": cfg.use_stop_loss,
            "use_entry_exit_signals": cfg.use_entry_exit_signals,
            "exit_no_momentum_days": cfg.exit_no_momentum_days,
            "initial_capital": cfg.initial_capital,
            "tranches": cfg.tranches,
            "max_positions": cfg.max_positions,
            "stop_loss_pct": cfg.stop_loss_pct,
            "equity_stop_pct": cfg.equity_stop_pct,
            "commission_bps": cfg.commission_bps,
            "tax_bps": cfg.tax_bps,
            "slippage_bps": cfg.slippage_bps,
            "kpi": cfg.kpi.__dict__ if cfg.kpi else None,
            "rotation_allow_daily": cfg.rotation_allow_daily,
            "rotation_delta_edge": cfg.rotation_delta_edge,
            "rotation_requires_fail": cfg.rotation_requires_fail,
            "rotation_edge_source": cfg.rotation_edge_source,
            "prob_field": cfg.prob_field,
            "entry_timing": cfg.entry_timing,
            "config_hash": cfg.config_hash,
            "config_snapshot": cfg.config_snapshot,
        },
        "summary": {
            "trades": len(trades),
            "win_rate": win_rate,
            "avg_trade_ret": avg_trade,
            "equity_end": eq[-1] if eq else float(cfg.initial_capital),
            "total_ret": ((eq[-1] - float(cfg.initial_capital)) / float(cfg.initial_capital)) if eq else 0.0,
            "max_drawdown": mdd,
            "cagr": cagr,
            "turnover": turnover,
            "cost_paid": total_cost,
            "halted": bool(halted),
        },
        "equity_curve": equity_curve,
        "trades": trades,
    }


def max_drawdown(equity: List[float]) -> float:
    peak = float("-inf")
    mdd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (x - peak) / peak if peak > 0 else 0.0
        if dd < mdd:
            mdd = dd
    return mdd


def _persist_backtest_result(cfg: BacktestConfig, result: dict, run_id: str) -> None:
    from utils.db import db_conn

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    trades = result.get("trades") or []
    equity_curve = result.get("equity_curve") or []
    with db_conn(commit_on_success=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME IN ('backtest_runs','backtest_trades','backtest_equity_curve')
                """,
            )
            present = {r["TABLE_NAME"] for r in (cur.fetchall() or [])}
        missing = {"backtest_runs", "backtest_trades", "backtest_equity_curve"} - present
        if missing:
            raise RuntimeError(
                "回測落庫表不存在："
                + ", ".join(sorted(missing))
                + "。請先套用 `migrations/20260115_backtest_tables.sql`（以及若有調整欄位則套用 `migrations/20260115_backtest_trades_entry_metadata.sql`）。"
            )
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO backtest_runs
                  (run_id, start_date, end_date, side, config_hash, config_snapshot, created_at)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    run_id,
                    cfg.start,
                    cfg.end,
                    cfg.side,
                    cfg.config_hash,
                    cfg.config_snapshot,
                    now,
                ),
            )

            if trades:
                rows = []
                for t in trades:
                    rows.append(
                        (
                            run_id,
                            t.get("stock_id"),
                            t.get("side"),
                            t.get("signal_date"),
                            t.get("entry_exec_date"),
                            t.get("entry_timing"),
                            t.get("entry_price"),
                            t.get("entry_score"),
                            t.get("entry_primary_strategy"),
                            json.dumps(t.get("entry_rationale_tags"), ensure_ascii=False),
                            t.get("entry_prob"),
                            t.get("exit_date"),
                            t.get("exit_price"),
                            t.get("ret_gross"),
                            t.get("ret_net"),
                            t.get("cost_paid"),
                            t.get("exit_reason"),
                            1 if t.get("kpi_passed") else 0,
                        )
                    )
                cur.executemany(
                    """
                    INSERT INTO backtest_trades
                      (run_id, stock_id, side, signal_date, entry_exec_date, entry_timing,
                       entry_price, entry_score, entry_primary_strategy, entry_rationale_tags, entry_prob,
                       exit_date, exit_price, ret_gross, ret_net, cost_paid, exit_reason, kpi_passed)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    """,
                    rows,
                )

            if equity_curve:
                rows = []
                for e in equity_curve:
                    rows.append(
                        (
                            run_id,
                            e.get("date"),
                            e.get("equity"),
                            e.get("drawdown"),
                            e.get("cash"),
                            e.get("positions_count") if e.get("positions_count") is not None else e.get("positions"),
                        )
                    )
                cur.executemany(
                    """
                    INSERT INTO backtest_equity_curve
                      (run_id, trading_date, equity, drawdown, cash, positions_count)
                    VALUES (%s,%s,%s,%s,%s,%s)
                    """,
                    rows,
                )


def run_backtest(cfg: BacktestConfig) -> dict:
    # holding_days=0 -> daily 模式（不固定持有天數）
    if int(cfg.holding_days) == 0:
        result = run_backtest_daily(cfg)
        run_id = uuid4().hex
        if cfg.persist_run:
            _persist_backtest_result(cfg, result, run_id)
        result["run_id"] = run_id
        return result

    cal = get_trading_calendar(cfg.start, cfg.end, cfg.calendar_stock_id)
    if len(cal) < (cfg.holding_days + 2):
        raise ValueError("交易日太少，請放大日期區間或換 calendar_stock_id")

    equity: List[float] = [1.0]
    equity_dates: List[date] = [cal[0]]
    trades: List[dict] = []

    i = 0
    while i < len(cal):
        signal_date = cal[i]
        entry_date = next_calendar_date(cal, signal_date, 1)
        max_exit_date = next_calendar_date(cal, signal_date, 1 + int(cfg.holding_days))
        if entry_date is None or max_exit_date is None:
            break

        day_rets: List[float] = []

        def run_side(side: str):
            rows = select_candidates(trading_date=signal_date, side=side, cfg=cfg, limit=cfg.top_n)

            for r in rows:
                sid = r["stock_id"]
                sl = r.get("stop_loss_price")
                sim = simulate_exit(
                    side=side,
                    stock_id=sid,
                    entry_date=entry_date,
                    max_exit_date=max_exit_date,
                    stop_loss_price=float(sl) if sl is not None else None,
                    use_stop_loss=bool(cfg.use_stop_loss),
                )
                if sim is None:
                    continue
                exit_date, exit_px, stopped, entry_px = sim

                if side == "long":
                    ret = (exit_px - entry_px) / entry_px
                else:
                    ret = (entry_px - exit_px) / entry_px

                day_rets.append(ret)
                trades.append(
                    {
                        "side": side,
                        "stock_id": sid,
                        "signal_date": fmt_date(signal_date),
                "entry_exec_date": fmt_date(entry_date),
                        "entry_exec_date": fmt_date(entry_date),
                        "entry_timing": "next_open",
                        "entry_primary_strategy": r.get("primary_strategy"),
                        "entry_rationale_tags": r.get("rationale_tags"),
                        "entry_score": float(r.get("score_long" if side == "long" else "score_short", r.get("score", 0)) or 0.0),
                        "entry_prob": r.get("entry_prob"),
                        "exit_date": fmt_date(exit_date),
                        "entry_price": float(entry_px),
                        "exit_price": float(exit_px),
                        "ret_gross": float(ret),
                        "ret_net": float(ret),
                        "cost_paid": 0.0,
                        "exit_reason": "time_exit",
                        "kpi_passed": False,
                        "kpi_failed": False,
                        "kpi_state": None,
                        "stopped": bool(stopped),
                    }
                )

        if cfg.side in ("long", "both"):
            run_side("long")
        if cfg.side in ("short", "both"):
            run_side("short")

        if day_rets:
            avg_ret = sum(day_rets) / len(day_rets)
            equity.append(equity[-1] * (1.0 + avg_ret))
        else:
            equity.append(equity[-1])
        equity_dates.append(max_exit_date)

        # 不重疊：往前跳 holding_days
        i += int(cfg.holding_days)

    total_ret = equity[-1] - 1.0
    wins = sum(1 for t in trades if float(t.get("ret_net", 0.0)) > 0)
    win_rate = (wins / len(trades)) if trades else 0.0
    avg_trade = (sum(float(t.get("ret_net", 0.0)) for t in trades) / len(trades)) if trades else 0.0
    mdd = max_drawdown(equity)

    result = {
        "config": {
            "start": fmt_date(cfg.start),
            "end": fmt_date(cfg.end),
            "side": cfg.side,
            "top_n": cfg.top_n,
            "holding_days": cfg.holding_days,
            "min_abs_score": cfg.min_abs_score,
            "entry_min_score": cfg.entry_min_score,
            "use_stop_loss": cfg.use_stop_loss,
            "use_entry_exit_signals": cfg.use_entry_exit_signals,
            "allowed_regimes_long": list(cfg.allowed_regimes_long),
            "allowed_regimes_short": list(cfg.allowed_regimes_short),
            "config_hash": cfg.config_hash,
            "config_snapshot": cfg.config_snapshot,
        },
        "summary": {
            "trades": len(trades),
            "win_rate": win_rate,
            "avg_trade_ret": avg_trade,
            "equity_end": equity[-1],
            "total_ret": total_ret,
            "max_drawdown": mdd,
        },
        "equity_curve": [{"date": fmt_date(d), "equity": float(v)} for d, v in zip(equity_dates, equity)],
        "trades": trades,
    }
    run_id = uuid4().hex
    if cfg.persist_run:
        _persist_backtest_result(cfg, result, run_id)
    result["run_id"] = run_id
    return result


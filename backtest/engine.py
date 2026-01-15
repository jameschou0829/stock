from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import settings
from utils.db import db_conn


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
    holding_days: int = 5
    min_abs_score: int = 0
    entry_min_score: int = 25
    use_stop_loss: bool = False
    use_entry_exit_signals: bool = True
    exit_no_momentum_days: int = 2
    calendar_stock_id: str = getattr(settings, "MARKET_PROXY_STOCK_ID", "0050")
    allowed_regimes_long: Tuple[str, ...] = ("bull", "range")
    allowed_regimes_short: Tuple[str, ...] = ("bear", "range")

    # ===== Phase A：可信回測參數 =====
    initial_capital: float = 1_000_000.0
    tranches: int = 10
    stop_loss_pct: float = 0.10
    equity_stop_pct: float = 0.30  # 累積虧損達 -30% 停止開新倉（仍管理既有倉）
    commission_bps: float = 14.25  # 0.1425%
    tax_bps: float = 30.0         # 0.30%（僅賣出收稅，short 進場視為賣出）
    slippage_bps: float = 5.0     # 0.05%


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


def get_signals_candidates_for_date(trading_date: date, side: str, top_n: int, min_abs_score: int) -> List[dict]:
    order = "DESC" if side == "long" else "ASC"
    with db_conn() as conn:
        with conn.cursor() as cur:
            if side == "long":
                cur.execute(
                    f"""
                    SELECT *
                    FROM stock_signals_v2
                    WHERE trading_date=%s AND score >= %s
                    ORDER BY score {order}
                    LIMIT %s
                    """,
                    (trading_date, int(min_abs_score), int(top_n)),
                )
            else:
                cur.execute(
                    f"""
                    SELECT *
                    FROM stock_signals_v2
                    WHERE trading_date=%s AND score <= %s
                    ORDER BY score {order}
                    LIMIT %s
                    """,
                    (trading_date, -int(min_abs_score), int(top_n)),
                )
            return cur.fetchall()


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
    這是偏保守的版本，目標是當動能消失就退出。
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


def evaluate_entry(row: dict, side: str, cfg: BacktestConfig) -> bool:
    regime = row.get("market_regime") or "unknown"
    score = int(row.get("score", 0) or 0)
    if side == "long":
        if regime not in cfg.allowed_regimes_long:
            return False
        if score < cfg.entry_min_score:
            return False
    else:
        if regime not in cfg.allowed_regimes_short:
            return False
        if score > -cfg.entry_min_score:
            return False
    return has_any_strategy(row, side)


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
    # 多抓一天：用於隔日開盤進場
    end_plus = next_calendar_date(cal, end0, 1) or end0

    with db_conn() as conn:
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
                """
                SELECT stock_id, trading_date, market_regime, score,
                       entry_long, entry_short, exit_long, exit_short,
                       primary_strategy,
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
    equity = float(cfg.initial_capital)
    tranche_value = initial / float(cfg.tranches)
    free_tranches = int(cfg.tranches)
    halted = False

    positions: Dict[str, dict] = {}  # stock_id -> pos
    no_mom_cnt: Dict[str, int] = {}
    trades: List[Trade] = []
    equity_curve = [{"date": fmt_date(cal[0]), "equity": float(equity)}]

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

    for idx in range(len(cal) - 1):
        d = cal[idx]
        d_next = cal[idx + 1]

        # 0) equity_stop 檢查（只影響開新倉）
        can_open_new_positions()

        # 1) 執行昨日產生的進場（在 d 的開盤進場）
        if pending_entry_signal_date == d and pending_entries and can_open_new_positions():
            bmap = bars_by_date.get(d, {})
            for sid in list(pending_entries):
                if sid in positions:
                    continue
                if free_tranches <= 0:
                    break
                b = bmap.get(sid)
                if not b:
                    continue
                entry_px = pick_price(b, "open")
                if entry_px is None or entry_px <= 0:
                    continue
                # side 由 signal 當天決定（強制 A 進 A 出：先只記 primary_strategy）
                sig0 = sig_by_date.get(cal[idx - 1], {}).get(sid) if idx > 0 else None
                side = None
                if sig0:
                    if int(sig0.get("entry_long", 0) or 0) == 1:
                        side = "long"
                    elif int(sig0.get("entry_short", 0) or 0) == 1:
                        side = "short"
                if side is None:
                    continue
                notional = tranche_value
                stop_px = _stop_price(float(entry_px), side=side, stop_loss_pct=float(cfg.stop_loss_pct))
                entry_cost = _trade_costs(
                    notional,
                    side=side,
                    action="entry",
                    commission_bps=cfg.commission_bps,
                    tax_bps=cfg.tax_bps,
                    slippage_bps=cfg.slippage_bps,
                )
                equity -= entry_cost
                positions[sid] = {
                    "side": side,
                    "signal_date": cal[idx - 1] if idx > 0 else d,
                    "entry_date": d,
                    "entry_px": float(entry_px),
                    "last_close": pick_price(b, "close") or float(entry_px),
                    "notional": float(notional),
                    "stop_px": float(stop_px),
                    "primary_strategy": sig0.get("primary_strategy") if sig0 else None,
                }
                no_mom_cnt[sid] = 0
                free_tranches -= 1
            pending_entries = set()

        # 2) 當天收盤：停損/出場 + 計算日 PnL（以 notional 加權）
        exits: set[str] = set()
        bmap_today = bars_by_date.get(d, {})
        sig_today_map = sig_by_date.get(d, {})

        day_pnl = 0.0
        for sid, pos in list(positions.items()):
            b = bmap_today.get(sid)
            if not b:
                continue
            cl = pick_price(b, "close")
            if cl is None or cl <= 0:
                continue

            side = pos["side"]
            notional = float(pos["notional"])

            # 日內停損（-10%）
            stop_px = float(pos["stop_px"])
            hit_px = _hit_stop(b, side=side, stop_px=stop_px) if cfg.use_stop_loss else None
            if hit_px is not None and pos.get("entry_date") != d:
                # 以 hit_px 出場，計算總損益（用 entry_px 基準）
                entry_px = float(pos["entry_px"])
                ret = (float(hit_px) - entry_px) / entry_px if side == "long" else (entry_px - float(hit_px)) / entry_px
                exit_notional = notional * (1.0 + ret)
                exit_cost = _trade_costs(
                    exit_notional,
                    side=side,
                    action="exit",
                    commission_bps=cfg.commission_bps,
                    tax_bps=cfg.tax_bps,
                    slippage_bps=cfg.slippage_bps,
                )
                equity += notional * ret
                equity -= exit_cost
                exits.add(sid)
                trades.append(
                    Trade(
                        side=side,
                        stock_id=sid,
                        signal_date=pos["signal_date"],
                        entry_date=pos["entry_date"],
                        exit_date=d,
                        entry_px=entry_px,
                        exit_px=float(hit_px),
                        ret=float(ret) - (exit_cost / notional if notional > 0 else 0.0),
                        stopped=True,
                    )
                )
                continue

            # 避免進場當天被洗出去（允許停損）
            if pos.get("entry_date") == d:
                pos["last_close"] = float(cl)
                continue

            sig = sig_today_map.get(sid)
            exit_flag = False
            if sig:
                exit_flag = int(sig.get("exit_long" if side == "long" else "exit_short", 0) or 0) == 1

            if sig and (not exit_flag):
                mom = has_momentum(sig, side)
                if not mom:
                    no_mom_cnt[sid] = int(no_mom_cnt.get(sid, 0)) + 1
                else:
                    no_mom_cnt[sid] = 0
                if int(no_mom_cnt[sid]) >= int(cfg.exit_no_momentum_days):
                    exit_flag = True

            # 日 PnL（close-to-close，以 notional 加權）
            prev_close = float(pos.get("last_close") or cl)
            r_day = (float(cl) - prev_close) / prev_close if side == "long" else (prev_close - float(cl)) / prev_close
            day_pnl += notional * r_day
            pos["last_close"] = float(cl)

            if exit_flag:
                # 用 close 出場，計入出場成本
                entry_px = float(pos["entry_px"])
                ret_total = (float(cl) - entry_px) / entry_px if side == "long" else (entry_px - float(cl)) / entry_px
                exit_notional = notional * (1.0 + ret_total)
                exit_cost = _trade_costs(
                    exit_notional,
                    side=side,
                    action="exit",
                    commission_bps=cfg.commission_bps,
                    tax_bps=cfg.tax_bps,
                    slippage_bps=cfg.slippage_bps,
                )
                equity += notional * ret_total
                equity -= exit_cost
                exits.add(sid)
                trades.append(
                    Trade(
                        side=side,
                        stock_id=sid,
                        signal_date=pos["signal_date"],
                        entry_date=pos["entry_date"],
                        exit_date=d,
                        entry_px=entry_px,
                        exit_px=float(cl),
                        ret=float(ret_total) - (exit_cost / notional if notional > 0 else 0.0),
                        stopped=False,
                    )
                )

        # 2.1 移除出場並歸還 tranche
        for sid in exits:
            positions.pop(sid, None)
            no_mom_cnt.pop(sid, None)
            free_tranches = min(int(cfg.tranches), free_tranches + 1)

        # 2.2 更新 equity（加上日內未實現 PnL）
        equity += float(day_pnl)
        equity_curve.append({"date": fmt_date(d), "equity": float(equity)})

        # 3) 產生下一交易日要進場的名單（用 d 當天 signals，d_next 開盤進）
        pending_entries = set()
        sig_map = sig_by_date.get(d, {})
        # 控制持倉上限：最多 top_n 檔（整體，不分多空）
        slots = max(0, int(cfg.top_n) - len(positions))
        if slots > 0 and can_open_new_positions() and free_tranches > 0:
            candidates: List[Tuple[str, int]] = []  # (sid, score)
            for sid, row in sig_map.items():
                if sid in positions:
                    continue
                if cfg.side == "long" and int(row.get("entry_long", 0) or 0) != 1:
                    continue
                if cfg.side == "short" and int(row.get("entry_short", 0) or 0) != 1:
                    continue
                if cfg.side == "both":
                    if int(row.get("entry_long", 0) or 0) != 1 and int(row.get("entry_short", 0) or 0) != 1:
                        continue
                # 仍保留舊 evaluate_entry 分數門檻（Phase A signals 會逐步切換）
                if cfg.side == "long" and not evaluate_entry(row, "long", cfg):
                    continue
                if cfg.side == "short" and not evaluate_entry(row, "short", cfg):
                    continue
                sc = int(row.get("score", 0) or 0)
                candidates.append((sid, sc))
            # 排序：多方取高分、空方取低分；both 用 abs(score) 大者優先
            if cfg.side == "short":
                candidates.sort(key=lambda x: x[1])
            elif cfg.side == "both":
                candidates.sort(key=lambda x: abs(x[1]), reverse=True)
            else:
                candidates.sort(key=lambda x: x[1], reverse=True)
            for sid, _ in candidates[:slots]:
                pending_entries.add(sid)

        pending_entry_signal_date = d_next

    # summary
    wins = sum(1 for t in trades if t.ret > 0)
    win_rate = (wins / len(trades)) if trades else 0.0
    avg_trade = (sum(t.ret for t in trades) / len(trades)) if trades else 0.0
    eq = [p["equity"] for p in equity_curve]
    mdd = max_drawdown(eq)
    return {
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
            "exit_no_momentum_days": cfg.exit_no_momentum_days,
            "initial_capital": cfg.initial_capital,
            "tranches": cfg.tranches,
            "stop_loss_pct": cfg.stop_loss_pct,
            "equity_stop_pct": cfg.equity_stop_pct,
            "commission_bps": cfg.commission_bps,
            "tax_bps": cfg.tax_bps,
            "slippage_bps": cfg.slippage_bps,
        },
        "summary": {
            "trades": len(trades),
            "win_rate": win_rate,
            "avg_trade_ret": avg_trade,
            "equity_end": eq[-1] if eq else float(cfg.initial_capital),
            "total_ret": ((eq[-1] - float(cfg.initial_capital)) / float(cfg.initial_capital)) if eq else 0.0,
            "max_drawdown": mdd,
            "halted": bool(halted),
        },
        "equity_curve": equity_curve,
        "trades": [
            {
                "side": t.side,
                "stock_id": t.stock_id,
                "signal_date": fmt_date(t.signal_date),
                "entry_date": fmt_date(t.entry_date),
                "exit_date": fmt_date(t.exit_date),
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "ret": t.ret,
                "stopped": t.stopped,
            }
            for t in trades
        ],
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


def run_backtest(cfg: BacktestConfig) -> dict:
    # holding_days=0 -> daily 模式（不固定持有天數）
    if int(cfg.holding_days) == 0:
        return run_backtest_daily(cfg)

    cal = get_trading_calendar(cfg.start, cfg.end, cfg.calendar_stock_id)
    if len(cal) < (cfg.holding_days + 2):
        raise ValueError("交易日太少，請放大日期區間或換 calendar_stock_id")

    equity: List[float] = [1.0]
    equity_dates: List[date] = [cal[0]]
    trades: List[Trade] = []

    i = 0
    while i < len(cal):
        signal_date = cal[i]
        entry_date = next_calendar_date(cal, signal_date, 1)
        max_exit_date = next_calendar_date(cal, signal_date, 1 + int(cfg.holding_days))
        if entry_date is None or max_exit_date is None:
            break

        day_rets: List[float] = []

        def run_side(side: str):
            rows = get_signals_candidates_for_date(signal_date, side, cfg.top_n, cfg.min_abs_score)

            # entry filter
            if cfg.use_entry_exit_signals:
                rows = [r for r in rows if evaluate_entry(r, side, cfg)]

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
                    Trade(
                        side=side,
                        stock_id=sid,
                        signal_date=signal_date,
                        entry_date=entry_date,
                        exit_date=exit_date,
                        entry_px=float(entry_px),
                        exit_px=float(exit_px),
                        ret=float(ret),
                        stopped=bool(stopped),
                    )
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
    wins = sum(1 for t in trades if t.ret > 0)
    win_rate = (wins / len(trades)) if trades else 0.0
    avg_trade = (sum(t.ret for t in trades) / len(trades)) if trades else 0.0
    mdd = max_drawdown(equity)

    return {
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
        "trades": [
            {
                "side": t.side,
                "stock_id": t.stock_id,
                "signal_date": fmt_date(t.signal_date),
                "entry_date": fmt_date(t.entry_date),
                "exit_date": fmt_date(t.exit_date),
                "entry_px": t.entry_px,
                "exit_px": t.exit_px,
                "ret": t.ret,
                "stopped": t.stopped,
            }
            for t in trades
        ],
    }


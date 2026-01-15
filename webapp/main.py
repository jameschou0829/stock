import os
import sys
from datetime import date
from typing import Optional

import csv
import io

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest.engine import BacktestConfig, run_backtest, parse_date
from configs import settings
from utils.db import db_conn
from backtest.engine import has_momentum


app = FastAPI(title="stock_test2 backtest")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_start": "2025-09-11",
            "default_end": "2026-01-09",
            "default_top_n": 20,
            "default_holding_days": 5,
            "default_min_abs_score": 10,
            "default_entry_min_score": getattr(settings, "SIGNALS_ENTRY_MIN_SCORE", 25),
        },
    )


@app.get("/stock/{stock_id}", response_class=HTMLResponse)
def stock_page(stock_id: str, request: Request):
    return templates.TemplateResponse(
        "stock.html",
        {
            "request": request,
            "stock_id": stock_id,
            "default_start": "2025-09-11",
            "default_end": "2026-01-09",
            "default_entry_min_score": getattr(settings, "SIGNALS_ENTRY_MIN_SCORE", 25),
        },
    )

def _get_latest_signals_date() -> Optional[date]:
    with db_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT MAX(trading_date) AS d FROM stock_signals_v2")
                row = cur.fetchone()
                return row["d"] if row and row.get("d") else None
        except Exception:
            # table 尚未建立/尚未產生 signals 時，避免首頁 500
            return None


def _has_column(conn, table_name: str, column_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND COLUMN_NAME = %s
            LIMIT 1
            """,
            (table_name, column_name),
        )
        return cur.fetchone() is not None


def _fmt_tags(sig: dict, side: str) -> list[str]:
    if side == "long":
        mapping = [
            ("strat_volume_momentum", "量大動能"),
            ("strat_price_volume_new_high", "價量創新高"),
            ("strat_breakout_edge", "突破邊緣"),
            ("strat_trust_breakout", "投信剛買準突破"),
            ("strat_trust_momentum_buy", "投信動能連買"),
            ("strat_foreign_big_buy", "外資剛大買"),
            ("strat_co_buy", "外資投信同買"),
        ]
    else:
        mapping = [
            ("strat_volume_momentum_weak", "量大動能(弱)"),
            ("strat_price_volume_new_low", "價量創新低"),
            ("strat_trust_breakdown", "投信剛賣準跌破"),
            ("strat_trust_momentum_sell", "投信動能連賣"),
            ("strat_foreign_big_sell", "外資剛大賣"),
            ("strat_co_sell", "外資投信同賣"),
        ]
    return [label for k, label in mapping if int(sig.get(k, 0) or 0) == 1]

def _fmt_rationale_tags(sig: dict, side: str) -> list[str]:
    """
    Web MVP：即使 DB 尚未有 rationale_tags 欄位，也能即時生成可讀原因。
    之後 Phase A 會把 rationale_tags 落到 DB，這裡仍可作為 fallback。
    """
    tags: list[str] = []
    if side == "long":
        if int(sig.get("is_foreign_first_buy", 0) or 0) == 1:
            tags.append("外資轉買第1天")
        if int(sig.get("is_trust_first_buy", 0) or 0) == 1:
            tags.append("投信轉買第1天")
        if int(sig.get("is_foreign_buy_3d", 0) or 0) == 1:
            tags.append("外資連買3日")
        if int(sig.get("is_trust_buy_3d", 0) or 0) == 1 or int(sig.get("trust_net_3d", 0) or 0) > 0:
            tags.append("投信近3日偏買")
        if int(sig.get("is_trust_buy_5d", 0) or 0) == 1:
            tags.append("投信連買5日")
        if int(sig.get("is_co_buy", 0) or 0) == 1:
            tags.append("土洋同買")
        if int(sig.get("above_ma20", 0) or 0) == 1:
            tags.append("收盤在MA20上")
        if int(sig.get("above_ma60", 0) or 0) == 1:
            tags.append("收盤在MA60上")
        if int(sig.get("is_price_breakout_20d", 0) or 0) == 1:
            tags.append("突破近20日新高")
        if int(sig.get("is_volume_breakout_20d", 0) or 0) == 1:
            tags.append("成交量創近20日新高")
        if int(sig.get("is_near_40d_high", 0) or 0) == 1:
            tags.append("接近40日高點（箱型上緣）")
        if int(sig.get("is_margin_risk", 0) or 0) == 1:
            tags.append("融資增加（風險）")
    else:
        if int(sig.get("is_foreign_first_sell", 0) or 0) == 1:
            tags.append("外資轉賣第1天")
        if int(sig.get("is_trust_first_sell", 0) or 0) == 1:
            tags.append("投信轉賣第1天")
        if int(sig.get("is_foreign_sell_3d", 0) or 0) == 1:
            tags.append("外資連賣3日")
        if int(sig.get("is_trust_sell_5d", 0) or 0) == 1:
            tags.append("投信連賣5日")
        if int(sig.get("is_co_sell", 0) or 0) == 1:
            tags.append("土洋同賣")
        # below MA：signals v2 沒有 below flag，改用 close/ma20 即時計算（若欄位存在）
        try:
            c = sig.get("close")
            m20 = sig.get("ma20")
            m60 = sig.get("ma60")
            if c is not None and m20 is not None and float(c) < float(m20):
                tags.append("收盤在MA20下")
            if c is not None and m60 is not None and float(c) < float(m60):
                tags.append("收盤在MA60下")
        except Exception:
            pass
        if int(sig.get("is_price_breakdown_20d", 0) or 0) == 1:
            tags.append("跌破近20日新低")
        if int(sig.get("is_volume_breakout_20d", 0) or 0) == 1:
            tags.append("下跌段放量（近20日量突破）")
        if int(sig.get("is_margin_risk", 0) or 0) == 1:
            tags.append("融資增加（風險）")
    return tags


def _get_stock_names(conn, stock_ids: list[str]) -> dict[str, str]:
    if not stock_ids:
        return {}
    placeholders = ",".join(["%s"] * len(stock_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT stock_id, stock_name
            FROM stock_info
            WHERE stock_id IN ({placeholders})
            """,
            tuple(stock_ids),
        )
        rows = cur.fetchall() or []
    return {r["stock_id"]: (r.get("stock_name") or "") for r in rows}


def _get_next_open_prices(conn, stock_ids: list[str], d: date) -> dict[str, float]:
    """
    entry_price 定義：T日產生訊號，T+1開盤進場 -> next trading day open
    這裡用單次 query 做 group-by + join，避免 N+1。
    """
    if not stock_ids:
        return {}
    placeholders = ",".join(["%s"] * len(stock_ids))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT t.stock_id, sd.open AS entry_open
            FROM (
              SELECT stock_id, MIN(trading_date) AS next_date
              FROM stock_daily
              WHERE trading_date > %s
                AND stock_id IN ({placeholders})
              GROUP BY stock_id
            ) t
            JOIN stock_daily sd
              ON sd.stock_id = t.stock_id AND sd.trading_date = t.next_date
            """,
            (d, *stock_ids),
        )
        rows = cur.fetchall() or []
    out: dict[str, float] = {}
    for r in rows:
        try:
            if r.get("entry_open") is None:
                continue
            out[r["stock_id"]] = float(r["entry_open"])
        except Exception:
            continue
    return out


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _compute_entry_streaks(
    *,
    conn,
    stock_ids: list[str],
    target_date: date,
    side: str,
    strategy: Optional[str],
    lookback_days: int = 60,
) -> dict[str, int]:
    """
    計算「上榜連續天數」：
    - 定義：entry_long/entry_short=1（若指定 strategy，則同時要求 strategy=1）
    - 連續：沿交易日往回，遇到第一天不符合就停止
    """
    if not stock_ids:
        return {}

    entry_col = "entry_long" if side == "long" else "entry_short"

    with conn.cursor() as cur:
        # 取最近 N 個交易日（用 signals table 的 trading_date）
        cur.execute(
            """
            SELECT DISTINCT trading_date
            FROM stock_signals_v2
            WHERE trading_date <= %s
            ORDER BY trading_date DESC
            LIMIT %s
            """,
            (target_date, int(lookback_days)),
        )
        dates_desc = [r["trading_date"] for r in (cur.fetchall() or [])]

        if not dates_desc or target_date not in set(dates_desc):
            return {sid: 0 for sid in stock_ids}

        # 一次抓齊：指定股票、指定日期集合
        placeholders_sid = ",".join(["%s"] * len(stock_ids))
        placeholders_d = ",".join(["%s"] * len(dates_desc))

        cols = [entry_col]
        if strategy:
            cols.append(strategy)
        cols_sql = ", ".join(cols)

        cur.execute(
            f"""
            SELECT stock_id, trading_date, {cols_sql}
            FROM stock_signals_v2
            WHERE stock_id IN ({placeholders_sid})
              AND trading_date IN ({placeholders_d})
            ORDER BY stock_id, trading_date DESC
            """,
            (*stock_ids, *dates_desc),
        )
        rows = cur.fetchall() or []

    by_stock: dict[str, dict[date, dict]] = {}
    for r in rows:
        by_stock.setdefault(r["stock_id"], {})[r["trading_date"]] = r

    streaks: dict[str, int] = {}
    for sid in stock_ids:
        cnt = 0
        for d in dates_desc:
            rr = by_stock.get(sid, {}).get(d)
            if not rr:
                break
            ok = int(rr.get(entry_col, 0) or 0) == 1
            if ok and strategy:
                ok = int(rr.get(strategy, 0) or 0) == 1
            if not ok:
                break
            cnt += 1
        streaks[sid] = cnt
    return streaks


@app.get("/entries", response_class=HTMLResponse)
def entries_page(request: Request):
    latest = _get_latest_signals_date()
    default_date = latest.strftime("%Y-%m-%d") if latest else "2026-01-13"
    return templates.TemplateResponse(
        "entries.html",
        {
            "request": request,
            "default_date": default_date,
            "default_side": "long",
            "default_limit": 50,
        },
    )


@app.get("/api/entries")
def api_entries(
    trading_date: str = Query(..., description="YYYY-MM-DD"),
    side: str = Query("long", pattern="^(long|short)$"),
    limit: int = Query(50, ge=1, le=500),
    strategy: Optional[str] = Query(None, description="strat_* 欄位名（例如 strat_trust_breakout）"),
):
    d = parse_date(trading_date)
    entry_col = "entry_long" if side == "long" else "entry_short"
    order = "DESC" if side == "long" else "ASC"

    allowed_strategies = {
        # long
        "strat_volume_momentum",
        "strat_price_volume_new_high",
        "strat_breakout_edge",
        "strat_trust_breakout",
        "strat_trust_momentum_buy",
        "strat_foreign_big_buy",
        "strat_co_buy",
        # short（先保留，頁面先做多也能用）
        "strat_volume_momentum_weak",
        "strat_price_volume_new_low",
        "strat_trust_breakdown",
        "strat_trust_momentum_sell",
        "strat_foreign_big_sell",
        "strat_co_sell",
    }
    if strategy is not None and strategy not in allowed_strategies:
        return {"date": trading_date, "side": side, "error": f"invalid strategy: {strategy}", "rows": []}

    with db_conn() as conn:
        streaks: dict[str, int] = {}
        # 若指定 strategy 但 DB 欄位還沒跟上（尚未跑過 build_signals.py 觸發 ALTER），回傳可理解的訊息
        if strategy:
            if not _has_column(conn, "stock_signals_v2", strategy):
                return {
                    "date": trading_date,
                    "side": side,
                    "strategy": strategy,
                    "error": f"DB 欄位不存在：{strategy}。請先執行 migration/重建 schema 後再查詢。",
                    "rows": [],
                }
        with conn.cursor() as cur:
            where = f"trading_date=%s AND {entry_col}=1"
            params = [d]
            if strategy:
                where += f" AND {strategy}=1"
            cur.execute(
                f"""
                SELECT *
                FROM stock_signals_v2
                WHERE {where}
                ORDER BY score {order}
                LIMIT %s
                """,
                (*params, int(limit)),
            )
            rows = cur.fetchall()

        stock_ids = [r["stock_id"] for r in (rows or []) if r.get("stock_id")]
        streaks = _compute_entry_streaks(
            conn=conn,
            stock_ids=stock_ids,
            target_date=d,
            side=side,
            strategy=strategy,
            lookback_days=60,
        )

    out = []
    for r in rows:
        close = _safe_float(r.get("close"))
        ma20 = _safe_float(r.get("ma20"))
        ma20_dist_pct = None
        if close is not None and ma20 is not None and ma20 > 0:
            ma20_dist_pct = (close / ma20 - 1.0) * 100.0

        out.append(
            {
                "stock_id": r["stock_id"],
                "trading_date": r["trading_date"].strftime("%Y-%m-%d") if r.get("trading_date") else None,
                "market_regime": r.get("market_regime"),
                "score": int(r.get("score", 0) or 0),
                "primary_strategy": r.get("primary_strategy"),
                "on_list_streak": int(streaks.get(r["stock_id"], 0)),
                "ma20_dist_pct": float(ma20_dist_pct) if ma20_dist_pct is not None else None,
                "dist_40d_high_pct": float(r["dist_40d_high_pct"]) if r.get("dist_40d_high_pct") is not None else None,
                "stop_loss_price": float(r["stop_loss_price"]) if r.get("stop_loss_price") is not None else None,
                "stop_loss_pct": float(r["stop_loss_pct"]) if r.get("stop_loss_pct") is not None else None,
                "tags": _fmt_tags(r, side),
            }
        )

    return {"date": trading_date, "side": side, "strategy": strategy, "count": len(out), "rows": out}

@app.get("/api/rankings")
def api_rankings(
    date: str = Query(..., description="YYYY-MM-DD"),
    side: str = Query("long", pattern="^(long|short)$"),
    limit: int = Query(50, ge=1, le=500),
    strategy: Optional[str] = Query(None, description="strat_* 欄位名（例如 strat_trust_breakout）"),
):
    """
    Web MVP：統一榜單 API。
    回傳 schema：
      symbol,name,date,side,score,primary_strategy,rationale_tags,entry_price,stop_price,entry_streak
    """
    d = parse_date(date)
    entry_col = "entry_long" if side == "long" else "entry_short"
    score_order = "DESC" if side == "long" else "ASC"

    allowed_strategies = {
        # long
        "strat_volume_momentum",
        "strat_price_volume_new_high",
        "strat_breakout_edge",
        "strat_trust_breakout",
        "strat_trust_momentum_buy",
        "strat_foreign_big_buy",
        "strat_co_buy",
        # short
        "strat_volume_momentum_weak",
        "strat_price_volume_new_low",
        "strat_trust_breakdown",
        "strat_trust_momentum_sell",
        "strat_foreign_big_sell",
        "strat_co_sell",
    }
    if strategy is not None and strategy not in allowed_strategies:
        return {"date": date, "side": side, "error": f"invalid strategy: {strategy}", "rows": []}

    with db_conn() as conn:
        if strategy:
            if not _has_column(conn, "stock_signals_v2", strategy):
                return {"date": date, "side": side, "strategy": strategy, "error": f"DB 欄位不存在：{strategy}", "rows": []}

        with conn.cursor() as cur:
            where = f"trading_date=%s AND {entry_col}=1"
            params = [d]
            if strategy:
                where += f" AND {strategy}=1"
            cur.execute(
                f"""
                SELECT *
                FROM stock_signals_v2
                WHERE {where}
                ORDER BY score {score_order}
                LIMIT %s
                """,
                (*params, int(limit)),
            )
            rows = cur.fetchall() or []

        stock_ids = [r["stock_id"] for r in rows if r.get("stock_id")]
        names = _get_stock_names(conn, stock_ids)
        entry_prices = _get_next_open_prices(conn, stock_ids, d)
        streaks = _compute_entry_streaks(conn=conn, stock_ids=stock_ids, target_date=d, side=side, strategy=strategy, lookback_days=120)

    out = []
    for r in rows:
        symbol = r.get("stock_id")
        if not symbol:
            continue
        rationale = r.get("rationale_tags")
        if isinstance(rationale, (list, tuple)):
            rationale_tags = list(rationale)
        else:
            rationale_tags = _fmt_rationale_tags(r, side)

        stop_price = r.get("stop_loss_price")
        if stop_price is None:
            try:
                c = float(r.get("close")) if r.get("close") is not None else None
                if c is not None:
                    stop_price = c * (1.0 - 0.10) if side == "long" else c * (1.0 + 0.10)
            except Exception:
                stop_price = None

        out.append(
            {
                "symbol": symbol,
                "name": names.get(symbol, ""),
                "date": r["trading_date"].strftime("%Y-%m-%d") if r.get("trading_date") else None,
                "side": side.upper(),
                "score": int(r.get("score", 0) or 0),
                "primary_strategy": r.get("primary_strategy"),
                "rationale_tags": rationale_tags,
                "entry_price": entry_prices.get(symbol),
                "stop_price": float(stop_price) if stop_price is not None else None,
                "entry_streak": int(streaks.get(symbol, 0)),
            }
        )
    return {"date": date, "side": side, "strategy": strategy, "count": len(out), "rows": out}


@app.get("/api/symbols/{symbol}/signals")
def api_symbol_signals(
    symbol: str,
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    side: str = Query("long", pattern="^(long|short)$"),
):
    """
    單一個股 signals 時序，回傳統一 schema（date 為每個 trading_date）。
    entry_price：僅在當天 entry_* = 1 時，回傳隔日開盤價。
    entry_streak：以該日為終點的連續 entry 天數（在此 symbol 範圍內可重現）。
    """
    s = parse_date(start)
    e = parse_date(end)
    entry_col = "entry_long" if side == "long" else "entry_short"

    with db_conn() as conn:
        # name
        name = ""
        with conn.cursor() as cur:
            cur.execute("SELECT stock_name FROM stock_info WHERE stock_id=%s", (symbol,))
            rr = cur.fetchone()
            name = (rr.get("stock_name") if rr else "") or ""

        # signals
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT *
                FROM stock_signals_v2
                WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (symbol, s, e),
            )
            sig_rows = cur.fetchall() or []

        # daily bars for entry_price（抓到 e 後一天，方便 shift）
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, open
                FROM stock_daily
                WHERE stock_id=%s AND trading_date BETWEEN %s AND DATE_ADD(%s, INTERVAL 10 DAY)
                ORDER BY trading_date
                """,
                (symbol, s, e),
            )
            bars = cur.fetchall() or []

    open_by_date = {b["trading_date"]: b.get("open") for b in bars if b.get("trading_date")}
    dates = [r["trading_date"] for r in sig_rows if r.get("trading_date")]
    # next trading date mapping（以 signals 的 trading_date 序列為準）
    next_date = {}
    for i in range(len(dates) - 1):
        next_date[dates[i]] = dates[i + 1]

    # entry_streak：在區間內由前往後累計
    streak = 0
    out = []
    for r in sig_rows:
        d = r.get("trading_date")
        if not d:
            continue
        is_entry = int(r.get(entry_col, 0) or 0) == 1
        if is_entry:
            streak += 1
        else:
            streak = 0

        entry_px = None
        if is_entry:
            nd = next_date.get(d)
            if nd:
                op = open_by_date.get(nd)
                if op is not None:
                    try:
                        entry_px = float(op)
                    except Exception:
                        entry_px = None

        stop_price = r.get("stop_loss_price")
        if stop_price is None:
            try:
                c = float(r.get("close")) if r.get("close") is not None else None
                if c is not None:
                    stop_price = c * (1.0 - 0.10) if side == "long" else c * (1.0 + 0.10)
            except Exception:
                stop_price = None

        out.append(
            {
                "symbol": symbol,
                "name": name,
                "date": d.strftime("%Y-%m-%d"),
                "side": side.upper(),
                "score": int(r.get("score", 0) or 0),
                "primary_strategy": r.get("primary_strategy"),
                "rationale_tags": _fmt_rationale_tags(r, side),
                "entry_price": entry_px,
                "stop_price": float(stop_price) if stop_price is not None else None,
                "entry_streak": int(streak),
            }
        )
    return {"symbol": symbol, "name": name, "side": side, "count": len(out), "rows": out}


@app.get("/api/backtest/summary")
def api_backtest_summary(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    side: str = Query("long", pattern="^(long|short|both)$"),
    top_n: int = Query(20, ge=1, le=200),
    holding_days: int = Query(5, ge=0, le=60),
    min_abs_score: int = Query(0, ge=0, le=200),
    entry_min_score: int = Query(25, ge=0, le=200),
    exit_no_momentum_days: int = Query(2, ge=1, le=200),
    use_stop_loss: bool = Query(True),
    use_entry_exit_signals: bool = Query(True),
    calendar_stock_id: str = Query(getattr(settings, "MARKET_PROXY_STOCK_ID", "0050")),
):
    cfg = BacktestConfig(
        start=parse_date(start),
        end=parse_date(end),
        side=side,
        top_n=int(top_n),
        holding_days=int(holding_days),
        min_abs_score=int(min_abs_score),
        entry_min_score=int(entry_min_score),
        use_stop_loss=bool(use_stop_loss),
        use_entry_exit_signals=bool(use_entry_exit_signals),
        exit_no_momentum_days=int(exit_no_momentum_days),
        calendar_stock_id=str(calendar_stock_id),
    )
    out = run_backtest(cfg)
    return {"config": out.get("config"), "summary": out.get("summary")}


@app.get("/api/backtest/trades")
def api_backtest_trades(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    side: str = Query("long", pattern="^(long|short|both)$"),
    top_n: int = Query(20, ge=1, le=200),
    holding_days: int = Query(5, ge=0, le=60),
    min_abs_score: int = Query(0, ge=0, le=200),
    entry_min_score: int = Query(25, ge=0, le=200),
    exit_no_momentum_days: int = Query(2, ge=1, le=200),
    use_stop_loss: bool = Query(True),
    use_entry_exit_signals: bool = Query(True),
    calendar_stock_id: str = Query(getattr(settings, "MARKET_PROXY_STOCK_ID", "0050")),
    format: str = Query("json", pattern="^(json|csv)$"),
):
    cfg = BacktestConfig(
        start=parse_date(start),
        end=parse_date(end),
        side=side,
        top_n=int(top_n),
        holding_days=int(holding_days),
        min_abs_score=int(min_abs_score),
        entry_min_score=int(entry_min_score),
        use_stop_loss=bool(use_stop_loss),
        use_entry_exit_signals=bool(use_entry_exit_signals),
        exit_no_momentum_days=int(exit_no_momentum_days),
        calendar_stock_id=str(calendar_stock_id),
    )
    out = run_backtest(cfg)
    trades = out.get("trades") or []

    if format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["side", "stock_id", "signal_date", "entry_date", "exit_date", "entry_px", "exit_px", "ret", "stopped"])
        for t in trades:
            w.writerow(
                [
                    t.get("side"),
                    t.get("stock_id"),
                    t.get("signal_date"),
                    t.get("entry_date"),
                    t.get("exit_date"),
                    t.get("entry_px"),
                    t.get("exit_px"),
                    t.get("ret"),
                    t.get("stopped"),
                ]
            )
        csv_text = buf.getvalue()
        filename = f"trades_{start}_{end}_{side}.csv"
        return Response(
            content=csv_text,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return {"config": out.get("config"), "count": len(trades), "trades": trades}

@app.get("/api/backtest")
def api_backtest(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    side: str = Query("long", pattern="^(long|short|both)$"),
    top_n: int = Query(20, ge=1, le=200),
    holding_days: int = Query(5, ge=0, le=60, description="0 代表不固定持有天數（用 exit/沒動能/停損 出場）"),
    min_abs_score: int = Query(0, ge=0, le=200),
    entry_min_score: int = Query(25, ge=0, le=200),
    exit_no_momentum_days: int = Query(2, ge=1, le=200),
    use_stop_loss: bool = Query(True),
    use_entry_exit_signals: bool = Query(True),
    calendar_stock_id: str = Query(getattr(settings, "MARKET_PROXY_STOCK_ID", "0050")),
):
    cfg = BacktestConfig(
        start=parse_date(start),
        end=parse_date(end),
        side=side,
        top_n=int(top_n),
        holding_days=int(holding_days),
        min_abs_score=int(min_abs_score),
        entry_min_score=int(entry_min_score),
        use_stop_loss=bool(use_stop_loss),
        use_entry_exit_signals=bool(use_entry_exit_signals),
        exit_no_momentum_days=int(exit_no_momentum_days),
        calendar_stock_id=str(calendar_stock_id),
    )
    return run_backtest(cfg)


@app.get("/api/stock_backtest")
def api_stock_backtest(
    stock_id: str = Query(...),
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    side: str = Query("long", pattern="^(long|short)$"),
    entry_min_score: int = Query(25, ge=0, le=200),
    exit_no_momentum_days: int = Query(2, ge=1, le=200),
    use_stop_loss: bool = Query(True),
    initial_capital: float = Query(1000000, gt=0, description="初始資金（用於計算總報酬%）"),
    position_sizing: str = Query("fixed", pattern="^(fixed|percent)$", description="部位大小：fixed=固定金額，percent=依資金比例（複利）"),
    entry_amount: float = Query(100000, gt=0, description="fixed 模式：每次進場金額"),
    entry_pct: float = Query(0.1, gt=0, le=1, description="percent 模式：每次投入資金比例（0~1）"),
    buy_fee_rate: float = Query(0.001425, ge=0, le=0.01, description="買進手續費率（預設 0.1425%）"),
    sell_fee_rate: float = Query(0.001425, ge=0, le=0.01, description="賣出手續費率（預設 0.1425%）"),
    sell_tax_rate: float = Query(0.003, ge=0, le=0.02, description="賣出交易稅率（預設 0.3%）"),
):
    """
    單一股票：回傳 K 棒 OHLC + 進出場 markers（以 signals 的 entry/exit + 沒動能 + 停損）
    進場：signal_date 符合 entry，下一交易日 open 進
    出場：每日檢查停損/exit/沒動能連續 N 天，觸發則當日 close 出（停損以 stop_loss_price）
    """
    s = parse_date(start)
    e = parse_date(end)

    # 多抓一段歷史用來計算 MA（避免區間一開始 MA 斷掉）
    lookback_start = s
    try:
        from datetime import timedelta
        lookback_start = s - timedelta(days=120)
    except Exception:
        pass

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trading_date, open, high, low, close
                FROM stock_daily
                WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (stock_id, lookback_start, e),
            )
            bars_all = cur.fetchall()
            cur.execute(
                """
                SELECT trading_date, score, market_regime,
                       entry_long, entry_short, exit_long, exit_short,
                       stop_loss_price,
                       close, ma20,
                       momentum_score, yesterday_turnover,
                       is_volume_breakout_20d, is_price_breakout_20d, is_price_breakdown_20d,
                       strat_volume_momentum, strat_price_volume_new_high, strat_trust_breakout, strat_trust_momentum_buy,
                       strat_foreign_big_buy, strat_co_buy,
                       strat_volume_momentum_weak, strat_price_volume_new_low, strat_trust_breakdown, strat_trust_momentum_sell,
                       strat_foreign_big_sell, strat_co_sell
                FROM stock_signals_v2
                WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
                ORDER BY trading_date
                """,
                (stock_id, lookback_start, e),
            )
            sig_rows_all = cur.fetchall()

    # 只顯示使用者指定區間
    bars = [b for b in (bars_all or []) if (b["trading_date"] >= s and b["trading_date"] <= e)]
    sig_rows = [r for r in (sig_rows_all or []) if (r["trading_date"] >= s and r["trading_date"] <= e)]

    if not bars:
        return {"ohlc": [], "ma": {}, "markers": [], "events": [], "trades": [], "summary": {}}

    bar_by_date = {b["trading_date"]: b for b in bars}
    sig_by_date = {r["trading_date"]: r for r in sig_rows}
    dates = [b["trading_date"] for b in bars]

    def bd(d):
        return {"year": d.year, "month": d.month, "day": d.day}

    # ===== MA 計算（用 close）=====
    def compute_sma_series(rows, window: int):
        # rows: bars_all，已按 trading_date asc
        closes = []
        out = {}
        for r in rows:
            c = r.get("close")
            closes.append(float(c) if c is not None else None)
        # rolling sum with None handling（遇到 None 就跳過該點）
        for i in range(len(rows)):
            if i + 1 < window:
                continue
            seg = closes[i + 1 - window : i + 1]
            if any(v is None for v in seg):
                continue
            out[rows[i]["trading_date"]] = sum(seg) / window
        # 只輸出顯示區間
        series = []
        for d in dates:
            v = out.get(d)
            if v is not None:
                series.append({"time": bd(d), "value": float(v)})
        return series

    ma = {
        "ma5": compute_sma_series(bars_all, 5),
        "ma10": compute_sma_series(bars_all, 10),
        "ma20": compute_sma_series(bars_all, 20),
        "ma60": compute_sma_series(bars_all, 60),
    }

    # state
    in_pos = False
    entry_date = None
    entry_px = None
    entry_notional = None
    entry_shares = None
    no_mom = 0

    markers = []
    events = []
    trades = []
    equity = float(initial_capital)

    def _fmt_tags(sig, side: str):
        if side == "long":
            mapping = [
                ("strat_volume_momentum", "量大動能"),
                ("strat_price_volume_new_high", "價量創新高"),
                ("strat_breakout_edge", "突破邊緣"),
                ("strat_trust_breakout", "投信剛買準突破"),
                ("strat_trust_momentum_buy", "投信動能連買"),
                ("strat_foreign_big_buy", "外資剛大買"),
                ("strat_co_buy", "外資投信同買"),
            ]
        else:
            mapping = [
                ("strat_volume_momentum_weak", "量大動能(弱)"),
                ("strat_price_volume_new_low", "價量創新低"),
                ("strat_trust_breakdown", "投信剛賣準跌破"),
                ("strat_trust_momentum_sell", "投信動能連賣"),
                ("strat_foreign_big_sell", "外資剛大賣"),
                ("strat_co_sell", "外資投信同賣"),
            ]
        tags = [label for k, label in mapping if int(sig.get(k, 0) or 0) == 1]
        return tags

    def entry_reason(sig):
        score = int(sig.get("score", 0) or 0)
        regime = sig.get("market_regime") or "unknown"
        tags = _fmt_tags(sig, side)
        tag_txt = "、".join(tags) if tags else "（無策略標籤）"
        return f"進場：{tag_txt}；分數={score}；市場={regime}；隔日開盤進"

    def exit_reason(sig, exit_flag: bool, no_mom_trigger: bool):
        reasons = []
        if exit_flag:
            reasons.append("出場訊號成立")
            # 粗略指出 build_signals 的 exit 條件之一是否命中
            try:
                close = sig.get("close")
                ma20v = sig.get("ma20")
                if close is not None and ma20v is not None and float(close) < float(ma20v):
                    reasons.append("收盤跌破 MA20")
            except Exception:
                pass
            if int(sig.get("is_price_breakdown_20d", 0) or 0) == 1:
                reasons.append("跌破近 20 日低點")
        if no_mom_trigger:
            reasons.append(f"動能消失連續 {exit_no_momentum_days} 天")
        return "；".join(reasons) or "出場"

    for i, d in enumerate(dates[:-1]):
        sig = sig_by_date.get(d)
        if not sig:
            continue

        # 進場判斷（用 entry_*，且再加一層 entry_min_score）
        if not in_pos:
            score = int(sig.get("score", 0) or 0)
            can_entry = False
            if side == "long":
                can_entry = int(sig.get("entry_long", 0) or 0) == 1 and score >= int(entry_min_score)
            else:
                can_entry = int(sig.get("entry_short", 0) or 0) == 1 and score <= -int(entry_min_score)

            if can_entry:
                d_next = dates[i + 1]
                b_next = bar_by_date.get(d_next)
                if b_next and b_next.get("open") is not None:
                    in_pos = True
                    entry_date = d_next
                    entry_px = float(b_next["open"])
                    # 計算本次投入金額（可複利）
                    if position_sizing == "percent":
                        notional = max(0.0, equity * float(entry_pct))
                    else:
                        notional = float(entry_amount)
                    notional = min(notional, equity)  # 不超過現金
                    shares = (notional / entry_px) if entry_px > 0 else 0.0
                    lots = shares / 1000.0

                    # 手續費/稅（簡化：不鎖住本金，只把費用從 equity 扣掉；損益在出場時計入）
                    if side == "long":
                        fee_in = notional * float(buy_fee_rate)
                        tax_in = 0.0
                    else:
                        # 放空進場是賣出，先扣賣出手續費+交易稅
                        fee_in = notional * float(sell_fee_rate)
                        tax_in = notional * float(sell_tax_rate)
                    equity -= (fee_in + tax_in)

                    entry_notional = float(notional)
                    entry_shares = float(shares)
                    no_mom = 0
                    markers.append({
                        "time": bd(d_next),
                        "position": "belowBar" if side == "long" else "aboveBar",
                        "color": "#10b981",
                        "shape": "arrowUp" if side == "long" else "arrowDown",
                        "text": "BUY" if side == "long" else "SELL",
                    })
                    events.append({
                        "time": bd(d_next),
                        "action": "BUY" if side == "long" else "SELL",
                        "price": float(entry_px),
                        "reason": entry_reason(sig),
                        "notional": float(notional),
                        "shares": float(shares),
                        "lots": float(lots),
                        "fee": float(fee_in),
                        "tax": float(tax_in),
                        "equity_after": float(equity),
                    })
            continue

        # 已持倉：在當天 bar 上檢查停損/出場（用 close/stop_loss）
        b = bar_by_date.get(d)
        if not b:
            continue
        lo = float(b["low"]) if b.get("low") is not None else None
        hi = float(b["high"]) if b.get("high") is not None else None
        cl = float(b["close"]) if b.get("close") is not None else None
        if cl is None:
            continue

        # 停損
        sl = sig.get("stop_loss_price")
        if use_stop_loss and sl is not None and lo is not None and hi is not None:
            sl = float(sl)
            hit = (side == "long" and lo <= sl) or (side == "short" and hi >= sl)
            if hit:
                op = float(b["open"]) if b.get("open") is not None else None
                # 跳空：若開盤已穿越停損，停損以開盤成交（避免不可能的「停損賣在開盤之上」）
                if side == "long":
                    exit_px = float(op) if (op is not None and op <= sl) else float(sl)
                else:
                    exit_px = float(op) if (op is not None and op >= sl) else float(sl)
                shares = float(entry_shares or 0.0)
                notional = float(entry_notional or 0.0)
                lots = shares / 1000.0

                if side == "long":
                    proceeds = shares * float(exit_px)
                    fee_out = proceeds * float(sell_fee_rate)
                    tax_out = proceeds * float(sell_tax_rate)
                    pnl = (proceeds - fee_out - tax_out) - notional
                else:
                    # short 出場是買回（買進手續費），交易稅已在進場賣出時扣
                    cost = shares * float(exit_px)
                    fee_out = cost * float(buy_fee_rate)
                    tax_out = 0.0
                    pnl = notional - (cost + fee_out)

                ret_net = (pnl / notional) if notional > 0 else 0.0
                equity += pnl
                trades.append({
                    "entry_date": bd(entry_date),
                    "exit_date": bd(d),
                    "side": side,
                    "entry_px": float(entry_px),
                    "exit_px": float(exit_px),
                    "ret": float(ret_net),
                    "pnl": float(pnl),
                    "notional": float(notional),
                    "shares": float(shares),
                    "lots": float(lots),
                    "fee_out": float(fee_out),
                    "tax_out": float(tax_out),
                    "reason": "停損觸發",
                    "equity_after": float(equity),
                })
                in_pos = False
                markers.append({
                    "time": bd(d),
                    "position": "aboveBar" if side == "long" else "belowBar",
                    "color": "#ef4444",
                    "shape": "arrowDown" if side == "long" else "arrowUp",
                    "text": "SL",
                })
                events.append({
                    "time": bd(d),
                    "action": "SL",
                    "price": float(exit_px),
                    "reason": f"停損：觸發 SL={sl:.2f}，以 {'開盤' if (op is not None and ((side=='long' and op<=sl) or (side=='short' and op>=sl))) else '停損價'} 成交",
                    "trade_ret": float(ret_net),
                    "pnl": float(pnl),
                    "notional": float(notional),
                    "shares": float(shares),
                    "lots": float(lots),
                    "equity_after": float(equity),
                })
                entry_date = None
                entry_px = None
                entry_notional = None
                entry_shares = None
                continue

        # 避免「進場當天」又被 exit/no_momentum 立刻洗出去（允許 SL）
        if entry_date == d:
            continue

        # exit_* 或沒動能連續 N 天
        exit_flag = int(sig.get("exit_long" if side == "long" else "exit_short", 0) or 0) == 1
        mom = has_momentum(sig, side)
        if not mom:
            no_mom += 1
        else:
            no_mom = 0

        no_mom_trigger = (no_mom >= int(exit_no_momentum_days))
        if exit_flag or no_mom_trigger:
            shares = float(entry_shares or 0.0)
            notional = float(entry_notional or 0.0)
            lots = shares / 1000.0

            if side == "long":
                proceeds = shares * float(cl)
                fee_out = proceeds * float(sell_fee_rate)
                tax_out = proceeds * float(sell_tax_rate)
                pnl = (proceeds - fee_out - tax_out) - notional
            else:
                cost = shares * float(cl)
                fee_out = cost * float(buy_fee_rate)
                tax_out = 0.0
                pnl = notional - (cost + fee_out)

            ret_net = (pnl / notional) if notional > 0 else 0.0
            equity += pnl
            trades.append({
                "entry_date": bd(entry_date),
                "exit_date": bd(d),
                "side": side,
                "entry_px": float(entry_px),
                "exit_px": float(cl),
                "ret": float(ret_net),
                "pnl": float(pnl),
                "notional": float(notional),
                "shares": float(shares),
                "lots": float(lots),
                "fee_out": float(fee_out),
                "tax_out": float(tax_out),
                "reason": exit_reason(sig, exit_flag, no_mom_trigger),
                "equity_after": float(equity),
            })
            in_pos = False
            markers.append({
                "time": bd(d),
                "position": "aboveBar" if side == "long" else "belowBar",
                "color": "#ef4444",
                "shape": "arrowDown" if side == "long" else "arrowUp",
                "text": "EXIT",
            })
            events.append({
                "time": bd(d),
                "action": "EXIT",
                "price": float(cl),
                "reason": exit_reason(sig, exit_flag, no_mom_trigger),
                "trade_ret": float(ret_net),
                "pnl": float(pnl),
                "notional": float(notional),
                "shares": float(shares),
                "lots": float(lots),
                "equity_after": float(equity),
            })
            entry_date = None
            entry_px = None
            entry_notional = None
            entry_shares = None
            continue

    ohlc = []
    for b in bars:
        d = b["trading_date"]
        if b.get("open") is None or b.get("high") is None or b.get("low") is None or b.get("close") is None:
            continue
        ohlc.append({
            "time": bd(d),
            "open": float(b["open"]),
            "high": float(b["high"]),
            "low": float(b["low"]),
            "close": float(b["close"]),
        })

    # summary
    trade_rets = [t["ret"] for t in trades]
    wins = sum(1 for r in trade_rets if r > 0)
    win_rate = (wins / len(trade_rets)) if trade_rets else 0.0
    avg_trade_ret = (sum(trade_rets) / len(trade_rets)) if trade_rets else 0.0
    total_ret = (equity - float(initial_capital)) / float(initial_capital)

    summary = {
        "initial_capital": float(initial_capital),
        "position_sizing": position_sizing,
        "entry_amount": float(entry_amount),
        "entry_pct": float(entry_pct),
        "buy_fee_rate": float(buy_fee_rate),
        "sell_fee_rate": float(sell_fee_rate),
        "sell_tax_rate": float(sell_tax_rate),
        "trades": len(trades),
        "win_rate": float(win_rate),
        "avg_trade_ret": float(avg_trade_ret),
        "equity_end": float(equity),
        "total_ret": float(total_ret),
    }

    return {"ohlc": ohlc, "ma": ma, "markers": markers, "events": events, "trades": trades, "summary": summary}


@app.get("/debug/volume-momentum")
def debug_volume_momentum(
    stock_id: str = Query(...),
    date: str = Query(..., description="YYYY-MM-DD"),
):
    """
    Debug endpoint：快速定位教學版「量大動能」策略所需欄位與中間值。
    """
    d = parse_date(date)
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT stock_id, trading_date,
                       turnover, yesterday_turnover,
                       momentum_score,
                       strat_volume_momentum,
                       close, volume,
                       ma20, ma60, atr14,
                       above_ma20, above_ma60,
                       is_price_breakout_20d, is_volume_breakout_20d
                FROM stock_signals_v2
                WHERE stock_id=%s AND trading_date=%s
                """,
                (stock_id, d),
            )
            row = cur.fetchone()

    if not row:
        return {"stock_id": stock_id, "date": date, "found": False}

    row["found"] = True
    return row


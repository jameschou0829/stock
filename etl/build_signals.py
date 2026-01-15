"""
etl/build_signals.py

根據「動能選股法」將每日資料整合成可解釋的強/弱勢訊號。

設計重點：
- 對齊目前 ETL 已產生的表：stock_daily / institution_trading / margin_trading / branch_trading
- 指標在此檔內計算（MA、突破、量能、法人連買），避免依賴 stock_daily 內的衍生欄位
- 產出寫入新的表 stock_signals_v2（避免跟舊版 schema 衝突）
"""

import sys, os, json
from datetime import date, timedelta, datetime
from statistics import mean
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import db_conn
from configs import settings
from configs.strategy_loader import load_strategy_config


# ============================================================
# 0. DB schema（v2）
# ============================================================

def _has_table(conn, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1
            FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
            LIMIT 1
            """,
            (table_name,),
        )
        return cur.fetchone() is not None


def require_signals_table(conn):
    """
    DDL 治理：禁止 runtime ALTER/CREATE。
    請先套用 migrations 內的 schema，再來執行 build_signals。
    """
    if not _has_table(conn, "stock_signals_v2"):
        raise RuntimeError("缺少資料表 stock_signals_v2，請先套用 migrations/*.sql 建立 schema。")


def _get_table_columns(conn, table_name: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
            """,
            (table_name,),
        )
        rows = cur.fetchall() or []
    return {str(r["COLUMN_NAME"]) for r in rows if r.get("COLUMN_NAME")}


# ============================================================
# 1. 工具：時間序列衍生
# ============================================================

def _sum_last_n(values_desc, n):
    return sum(values_desc[:n]) if values_desc else 0

def _safe_mean(values):
    return mean(values) if values else None


def _mean_if_all_present(values):
    """
    對 None/缺值敏感的 mean：只要窗口內含 None，就視為不可計算（回 None）
    這符合你要求的「區間內有 None -> 該日指標不可計算」。
    """
    if not values:
        return None
    if any(v is None for v in values):
        return None
    return mean(values)


def _max_if_all_present(values):
    if not values:
        return None
    if any(v is None for v in values):
        return None
    return max(values)


def _min_if_all_present(values):
    if not values:
        return None
    if any(v is None for v in values):
        return None
    return min(values)


def calc_momentum_score(
    close: Optional[float],
    ma20: Optional[float],
    atr14: Optional[float],
    is_price_breakout_20d: int = 0,
) -> float:
    """
    教學版策略要用 momentum_score > 7.5 做門檻，因此需要「能落地、可解釋」且數值大致落在 0~20 的設計。

    本專案採用：
      momentum_score = clamp( ATR% + 趨勢偏離 + 突破加分 , 0, 20 )

    - ATR% = (ATR14 / close) * 100
      代表波動幅度占股價的百分比（常見約 1~8，個股會更高）
    - 趨勢偏離 = max(0, close/ma20 - 1) * 100
      只看多方「站上 MA20 的偏離幅度」（避免下跌也被算成動能）
    - 突破加分：若 20 日新高突破，額外加 3 分

    注意：任何 None/0 會回傳 0.0，確保不因缺值或除以 0 炸裂。
    """
    try:
        c = float(close) if close is not None else None
        m20 = float(ma20) if ma20 is not None else None
        atr = float(atr14) if atr14 is not None else None
    except Exception:
        return 0.0

    if c is None or c <= 0:
        return 0.0

    atr_pct = (atr / c) * 100.0 if (atr is not None and atr >= 0) else 0.0
    trend = 0.0
    if m20 is not None and m20 > 0:
        trend = max(0.0, (c / m20) - 1.0) * 100.0

    bonus = 3.0 if int(is_price_breakout_20d or 0) == 1 else 0.0
    score = atr_pct + trend + bonus
    # clamp to 0~20
    if score < 0:
        score = 0.0
    if score > 20:
        score = 20.0
    return float(score)


def calc_teaching_strat_volume_momentum(momentum_score, yesterday_turnover, *, strategy_cfg: Optional[dict] = None) -> int:
    """
    教學版：多方策略｜量大動能（多方）
    上榜條件（同時成立）：
      1) momentum_score > threshold
      2) yesterday_turnover > threshold
    缺值視為 0，避免因 None/型別問題炸裂。
    """
    if strategy_cfg is None:
        strategy_cfg, _, _ = load_strategy_config({})
    thresholds = strategy_cfg.get("scoring", {}).get("thresholds", {})
    momentum_th = float(thresholds.get("momentum_score_high", 0) or 0.0)
    turnover_th = int(thresholds.get("turnover_min", 0) or 0)
    try:
        ms = float(momentum_score or 0.0)
    except Exception:
        ms = 0.0
    try:
        yto = int(yesterday_turnover or 0)
    except Exception:
        yto = 0
    return 1 if (ms > momentum_th and yto > turnover_th) else 0


# ============================================================
# 2. 大盤天氣圖（用 proxy 近似）
# ============================================================

def calc_market_regime(proxy_daily_desc):
    """
    步驟1：大盤天氣圖（偏多/震盪/偏空）
    這裡用單一 proxy（預設 0050）近似市場狀態。
    """
    # ma60 需要至少 60 筆 close，另外需要 ma20_prev
    if len(proxy_daily_desc) < 61:
        return "unknown"

    closes = [r["close"] for r in reversed(proxy_daily_desc)]  # 舊→新
    close = closes[-1]
    # 任一缺值 -> regime 不可判斷（避免 mean(None) TypeError）
    ma20 = _mean_if_all_present(closes[-20:])
    ma60 = _mean_if_all_present(closes[-60:])
    ma20_prev = _mean_if_all_present(closes[-21:-1])  # 昨日 ma20

    if close is None or ma20 is None or ma60 is None or ma20_prev is None:
        return "unknown"

    ma20_up = (ma20 - ma20_prev) > 0

    if close > ma20 > ma60 and ma20_up:
        return "bull"
    if close < ma20 < ma60 and (not ma20_up):
        return "bear"
    return "range"


# ============================================================
# 3. K 棒（日 K）訊號（內建計算）
# ============================================================

def calc_kbar_signals(daily_desc):
    """
    daily_desc: 由新到舊排序，至少 60 天
    """
    if not daily_desc:
        return {
            "close": None,
            "volume": None,
            "ma20": None,
            "ma60": None,
            "above_ma20": 0,
            "above_ma60": 0,
            "is_price_breakout_20d": 0,
            "is_price_breakdown_20d": 0,
            "is_volume_breakout_20d": 0,
            "dist_40d_high_pct": None,
            "is_near_40d_high": 0,
            "is_price_breakout_400d": 0,
            "is_volume_breakout_10d": 0,
        }

    # 舊→新（注意：任何缺值都要能安全處理，不得 TypeError）
    closes = [float(r["close"]) if r.get("close") is not None else None for r in reversed(daily_desc)]
    vols = [int(r["volume"]) if r.get("volume") is not None else None for r in reversed(daily_desc)]
    highs = [float(r["high"]) if r.get("high") is not None else None for r in reversed(daily_desc)]
    lows = [float(r["low"]) if r.get("low") is not None else None for r in reversed(daily_desc)]

    # 今日任一關鍵欄位缺值（close/high/low/volume）或 close<=0 -> 當日指標視為不可計算
    if closes[-1] is None or closes[-1] <= 0 or vols[-1] is None or highs[-1] is None or lows[-1] is None:
        return {
            "close": closes[-1] if closes else None,
            "volume": vols[-1] if vols else None,
            "ma20": None,
            "ma60": None,
            "atr14": None,
            "above_ma20": 0,
            "above_ma60": 0,
            "is_price_breakout_20d": 0,
            "is_price_breakdown_20d": 0,
            "is_volume_breakout_20d": 0,
            "dist_40d_high_pct": None,
            "is_near_40d_high": 0,
            "is_price_breakout_400d": 0,
            "is_volume_breakout_10d": 0,
        }

    close = closes[-1]
    vol = vols[-1]
    # MA：窗口內含 None 或長度不足 -> 不可計算（回 None）
    ma20 = _mean_if_all_present(closes[-20:]) if len(closes) >= 20 else None
    ma60 = _mean_if_all_present(closes[-60:]) if len(closes) >= 60 else None

    # 20日新高/新低（不含今天）
    prev_20 = closes[-21:-1] if len(closes) >= 21 else []
    prev_20_max = _max_if_all_present(prev_20)
    prev_20_min = _min_if_all_present(prev_20)

    is_breakout = 1 if (prev_20_max is not None and close > prev_20_max) else 0
    is_breakdown = 1 if (prev_20_min is not None and close < prev_20_min) else 0

    # 20日最大量（不含今天）
    prev_20_vol = vols[-21:-1] if len(vols) >= 21 else []
    prev_20_vol_max = _max_if_all_present(prev_20_vol)
    is_vol_breakout = 1 if (prev_20_vol_max is not None and vol > prev_20_vol_max) else 0

    # 10日最大量（不含今天）— 對應「成交量創 10 日新高」
    prev_10_vol = vols[-11:-1] if len(vols) >= 11 else []
    prev_10_vol_max = _max_if_all_present(prev_10_vol)
    is_vol_breakout_10d = 1 if (prev_10_vol_max is not None and vol > prev_10_vol_max) else 0

    # 400日新高（不含今天）— 對應「價創 400 日新高」
    prev_400_close = closes[-401:-1] if len(closes) >= 401 else []
    prev_400_max_close = _max_if_all_present(prev_400_close)
    is_breakout_400d = 1 if (prev_400_max_close is not None and close > prev_400_max_close) else 0

    # 40日最高價乖離（策略2：投信剛買準突破）
    dist_40d_high_pct = None
    is_near_40d_high = 0
    highs_40 = highs[-40:] if len(highs) >= 40 else []
    high_40 = _max_if_all_present(highs_40)
    if high_40 is not None and high_40 > 0 and close is not None and close > 0:
        dist_40d_high_pct = ((float(close) / float(high_40)) - 1.0) * 100.0
        # PDF：40日最高價乖離介於(含)2%~-2%
        if -2.0 <= dist_40d_high_pct <= 2.0:
            is_near_40d_high = 1

    # ATR14（需要 high/low/prev_close）
    atr14 = None
    # 需要至少 15 筆（含 prev_close），且窗口內不得有 None
    if len(closes) >= 15 and len(highs) >= 15 and len(lows) >= 15:
        window_highs = highs[-15:]
        window_lows = lows[-15:]
        window_closes = closes[-15:]  # 含今天；prev_close 需要 i-1，所以也要求 closes[-15] 不為 None
        if not (any(v is None for v in window_highs) or any(v is None for v in window_lows) or any(v is None for v in window_closes)):
            trs = []
            for i in range(len(closes) - 14, len(closes)):  # 最後 14 天
                hi = float(highs[i])
                lo = float(lows[i])
                prev_close = float(closes[i - 1])
                tr = max(hi - lo, abs(hi - prev_close), abs(lo - prev_close))
                trs.append(tr)
            atr14 = _mean_if_all_present(trs)

    return {
        "close": close,
        "volume": vol,
        "ma20": ma20,
        "ma60": ma60,
        "atr14": atr14,
        "above_ma20": 1 if (ma20 is not None and close > ma20) else 0,
        "above_ma60": 1 if (ma60 is not None and close > ma60) else 0,
        "is_price_breakout_20d": is_breakout,
        "is_price_breakdown_20d": is_breakdown,
        "is_volume_breakout_20d": is_vol_breakout,
        "dist_40d_high_pct": dist_40d_high_pct,
        "is_near_40d_high": is_near_40d_high,
        "is_price_breakout_400d": is_breakout_400d,
        "is_volume_breakout_10d": is_vol_breakout_10d,
    }


# ============================================================
# 4. 法人（外資、投信、自營）
# ============================================================

def calc_institution_signals(inst_desc):
    """
    inst_desc: 由新到舊排序，至少 5 天（含今天）
    """
    if not inst_desc:
        return {
            "foreign_net": 0,
            "trust_net": 0,
            "dealer_net": 0,
            "foreign_net_3d": 0,
            "trust_net_3d": 0,
            "trust_net_5d": 0,
            "trust_buy_streak": 0,
            "is_foreign_first_buy": 0,
            "is_trust_first_buy": 0,
            "is_co_buy": 0,
            "is_foreign_buy_3d": 0,
            "is_trust_buy_3d": 0,
            "is_trust_buy_5d": 0,
            "is_foreign_first_sell": 0,
            "is_trust_first_sell": 0,
            "is_foreign_sell_3d": 0,
            "is_trust_sell_5d": 0,
            "is_co_sell": 0,
        }

    today = inst_desc[0]
    foreign_today = int(today.get("foreign_net", 0))
    trust_today = int(today.get("trust_net", 0))
    dealer_today = int(today.get("dealer_net", 0))

    foreign_net_3d = _sum_last_n([int(r.get("foreign_net", 0)) for r in inst_desc], 3)
    trust_net_3d = _sum_last_n([int(r.get("trust_net", 0)) for r in inst_desc], 3)
    trust_net_5d = _sum_last_n([int(r.get("trust_net", 0)) for r in inst_desc], 5)

    # 投信連買天數（從今天往回連續 >0 的天數）
    trust_buy_streak = 0
    for r in inst_desc:
        if int(r.get("trust_net", 0)) > 0:
            trust_buy_streak += 1
        else:
            break

    # first buy：昨天(或最近一天)為 <=0，今天轉正
    y_foreign = int(inst_desc[1].get("foreign_net", 0)) if len(inst_desc) > 1 else 0
    y_trust = int(inst_desc[1].get("trust_net", 0)) if len(inst_desc) > 1 else 0

    is_foreign_first_buy = 1 if (foreign_today > 0 and y_foreign <= 0) else 0
    is_trust_first_buy = 1 if (trust_today > 0 and y_trust <= 0) else 0
    is_foreign_first_sell = 1 if (foreign_today < 0 and y_foreign >= 0) else 0
    is_trust_first_sell = 1 if (trust_today < 0 and y_trust >= 0) else 0

    is_foreign_buy_3d = 1 if all(int(r.get("foreign_net", 0)) > 0 for r in inst_desc[:3]) else 0
    # PDF：投信近 3 日內買超（用近 3 日合計 > 0 作為可落地的近似）
    is_trust_buy_3d = 1 if trust_net_3d > 0 else 0
    is_trust_buy_5d = 1 if len(inst_desc) >= 5 and all(int(r.get("trust_net", 0)) > 0 for r in inst_desc[:5]) else 0
    is_foreign_sell_3d = 1 if all(int(r.get("foreign_net", 0)) < 0 for r in inst_desc[:3]) else 0
    is_trust_sell_5d = 1 if len(inst_desc) >= 5 and all(int(r.get("trust_net", 0)) < 0 for r in inst_desc[:5]) else 0

    is_co_buy = 1 if (foreign_today > 0 and trust_today > 0) else 0
    is_co_sell = 1 if (foreign_today < 0 and trust_today < 0) else 0

    return {
        "foreign_net": foreign_today,
        "trust_net": trust_today,
        "dealer_net": dealer_today,
        "foreign_net_3d": foreign_net_3d,
        "trust_net_3d": trust_net_3d,
        "trust_net_5d": trust_net_5d,
        "trust_buy_streak": trust_buy_streak,
        "is_foreign_first_buy": is_foreign_first_buy,
        "is_trust_first_buy": is_trust_first_buy,
        "is_co_buy": is_co_buy,
        "is_foreign_buy_3d": is_foreign_buy_3d,
        "is_trust_buy_3d": is_trust_buy_3d,
        "is_trust_buy_5d": is_trust_buy_5d,
        "is_foreign_first_sell": is_foreign_first_sell,
        "is_trust_first_sell": is_trust_first_sell,
        "is_foreign_sell_3d": is_foreign_sell_3d,
        "is_trust_sell_5d": is_trust_sell_5d,
        "is_co_sell": is_co_sell,
    }


# ============================================================
# 5. 分點（簡化集中度）
# ============================================================

def calc_branch_signals(branch_rows, *, concentration_ratio: float = 0.0):
    """
    你目前的 branch_trading 沒有「主力」標記，因此用集中度做替代：
    - top5_buy_ratio：前五大分點買量集中度
    - top5_net_ratio：前五大分點淨買集中度（用 abs(net) 做分母）
    """
    if not branch_rows:
        return {
            "branch_top5_buy_ratio": None,
            "branch_top5_net_ratio": None,
            "is_branch_concentration": 0,
        }

    buys = [int(r.get("buy", 0)) for r in branch_rows]
    nets = [abs(int(r.get("net", 0))) for r in branch_rows]

    total_buy = sum(buys)
    total_net_abs = sum(nets)

    top5_by_buy = sorted(branch_rows, key=lambda x: int(x.get("buy", 0)), reverse=True)[:5]
    top5_buy = sum(int(r.get("buy", 0)) for r in top5_by_buy)
    buy_ratio = (top5_buy / total_buy) if total_buy > 0 else 0.0

    top5_by_net = sorted(branch_rows, key=lambda x: abs(int(x.get("net", 0))), reverse=True)[:5]
    top5_net_abs = sum(abs(int(r.get("net", 0))) for r in top5_by_net)
    net_ratio = (top5_net_abs / total_net_abs) if total_net_abs > 0 else 0.0

    # 經驗閾值：買量集中度門檻由 strategy.yaml 控制
    is_concentration = 1 if buy_ratio >= float(concentration_ratio) else 0

    return {
        "branch_top5_buy_ratio": buy_ratio,
        "branch_top5_net_ratio": net_ratio,
        "is_branch_concentration": is_concentration,
    }


# ============================================================
# 6. 融資（風險）
# ============================================================

def calc_margin_signals(margin_desc):
    """
    margin_desc: 由新到舊排序，至少 4 天（用來算 3d 變化）
    """
    if not margin_desc:
        return {
            "margin_balance": None,
            "margin_balance_chg_3d": None,
            "is_margin_risk": 0,
        }

    bal_today = int(margin_desc[0].get("margin_balance", 0))
    bal_3d_ago = int(margin_desc[3].get("margin_balance", 0)) if len(margin_desc) >= 4 else bal_today
    chg_3d = bal_today - bal_3d_ago

    # 融資增加視為風險（偏投機/槓桿升高），先做保守扣分
    is_risk = 1 if chg_3d > 0 else 0

    return {
        "margin_balance": bal_today,
        "margin_balance_chg_3d": chg_3d,
        "is_margin_risk": is_risk,
    }


# ============================================================
# 7. 策略（步驟2）與總分
# ============================================================

def calc_strategies_and_score(market_regime, sig, strategy_cfg: dict):
    """
    以講義「策略上榜」的概念，將條件拆成明確策略旗標，再由旗標與基礎條件給分。
    所有門檻/係數均由 strategy_cfg 提供。
    """
    scoring = strategy_cfg.get("scoring", {}) if isinstance(strategy_cfg, dict) else {}
    points = scoring.get("points", {})
    strat_points = scoring.get("strategy_points", {})
    thresholds = scoring.get("thresholds", {})
    regime_weights = scoring.get("regime_weights", {})
    allowed_regimes_long = scoring.get("allowed_regimes_long", ["bull", "range"])
    allowed_regimes_short = scoring.get("allowed_regimes_short", ["bear", "range"])
    entry_min_score_long = int(scoring.get("entry_min_score_long", 0) or 0)
    entry_min_score_short = int(scoring.get("entry_min_score_short", 0) or 0)
    entry_min_score_range_boost = int(scoring.get("entry_min_score_range_boost", 0) or 0)

    detail_long = {}
    detail_short = {}
    score_long = 0
    score_short = 0
    rationale = set()

    close = sig.get("close")
    ma20 = sig.get("ma20")
    ma60 = sig.get("ma60")

    # --------- 技術面（步驟3）---------
    if sig.get("above_ma20"):
        w = int(points.get("above_ma20", 0) or 0)
        score_long += w
        detail_long["above_ma20"] = w
        rationale.add("above_ma20")
    if sig.get("above_ma60"):
        w = int(points.get("above_ma60", 0) or 0)
        score_long += w
        detail_long["above_ma60"] = w
        rationale.add("above_ma60")
    if sig.get("is_price_breakout_20d"):
        w = int(points.get("breakout_20d", 0) or 0)
        score_long += w
        detail_long["breakout_20d"] = w
        rationale.add("breakout_20d")
    if sig.get("is_volume_breakout_20d"):
        w = int(points.get("volume_breakout_20d_long", 0) or 0)
        score_long += w
        detail_long["volume_breakout_20d"] = w
        rationale.add("volume_breakout_20d")

    # 空方技術面（分開計分，不用負分混在同一 score）
    below_ma20 = (close is not None and ma20 is not None and close < ma20)
    below_ma60 = (close is not None and ma60 is not None and close < ma60)
    if below_ma20:
        w = int(points.get("below_ma20", 0) or 0)
        score_short += w
        detail_short["below_ma20"] = w
        rationale.add("below_ma20")
    if below_ma60:
        w = int(points.get("below_ma60", 0) or 0)
        score_short += w
        detail_short["below_ma60"] = w
        rationale.add("below_ma60")
    if sig.get("is_price_breakdown_20d"):
        w = int(points.get("breakdown_20d", 0) or 0)
        score_short += w
        detail_short["breakdown_20d"] = w
        rationale.add("breakdown_20d")
    if sig.get("is_volume_breakout_20d"):
        # 下跌段爆量常見，空方也給分（與多方不同權重）
        w = int(points.get("volume_breakout_20d_short", 0) or 0)
        score_short += w
        detail_short["volume_breakout_20d"] = w

    # --------- 籌碼面：法人（步驟4）---------
    if sig.get("is_foreign_first_buy"):
        w = int(points.get("foreign_first_buy", 0) or 0)
        score_long += w
        detail_long["foreign_buy_day1"] = w
        rationale.add("foreign_buy_day1")
    if sig.get("is_trust_first_buy"):
        w = int(points.get("trust_first_buy", 0) or 0)
        score_long += w
        detail_long["trust_buy_day1"] = w
        rationale.add("trust_buy_day1")
    if sig.get("is_foreign_buy_3d"):
        w = int(points.get("foreign_buy_3d", 0) or 0)
        score_long += w
        detail_long["foreign_buy_3d"] = w
        rationale.add("foreign_buy_3d")
    if sig.get("is_trust_buy_5d"):
        w = int(points.get("trust_buy_5d", 0) or 0)
        score_long += w
        detail_long["trust_buy_5d"] = w
        rationale.add("trust_buy_5d")
    if sig.get("is_co_buy"):
        w = int(points.get("co_buy", 0) or 0)
        score_long += w
        detail_long["co_buy_today"] = w
        rationale.add("co_buy_today")

    if sig.get("is_foreign_first_sell"):
        w = int(points.get("foreign_first_sell", 0) or 0)
        score_short += w
        detail_short["foreign_sell_day1"] = w
        rationale.add("foreign_sell_day1")
    if sig.get("is_trust_first_sell"):
        w = int(points.get("trust_first_sell", 0) or 0)
        score_short += w
        detail_short["trust_sell_day1"] = w
        rationale.add("trust_sell_day1")
    if sig.get("is_foreign_sell_3d"):
        w = int(points.get("foreign_sell_3d", 0) or 0)
        score_short += w
        detail_short["foreign_sell_3d"] = w
        rationale.add("foreign_sell_3d")
    if sig.get("is_trust_sell_5d"):
        w = int(points.get("trust_sell_5d", 0) or 0)
        score_short += w
        detail_short["trust_sell_5d"] = w
        rationale.add("trust_sell_5d")
    if sig.get("is_co_sell"):
        w = int(points.get("co_sell", 0) or 0)
        score_short += w
        detail_short["co_sell_today"] = w
        rationale.add("co_sell_today")

    # --------- 分點集中度（偏輔助）---------
    if sig.get("is_branch_concentration"):
        w = int(points.get("branch_concentration", 0) or 0)
        score_long += w
        detail_long["branch_concentration"] = w
        rationale.add("branch_concentration")

    # --------- 融資風險（步驟5前的風險提示）---------
    # 先做保守版：只扣多方（避免把放空也一起削弱）
    if sig.get("is_margin_risk"):
        w = int(points.get("margin_risk_penalty", 0) or 0)
        score_long += w
        detail_long["margin_risk"] = w
        rationale.add("margin_risk")

    # --------- 策略旗標（步驟2）---------
    try:
        yto = int(sig.get("yesterday_turnover") or 0)
    except Exception:
        yto = 0
    try:
        t3 = int(sig.get("trust_net_3d") or 0)
    except Exception:
        t3 = 0

    turnover_min = thresholds.get("turnover_min", 0) or 0
    momentum_score_high = float(thresholds.get("momentum_score_high", 0) or 0.0)
    momentum_score_medium = float(thresholds.get("momentum_score_medium", 0) or 0.0)
    trust_buy_streak_min = int(thresholds.get("trust_buy_streak_min", 0) or 0)
    foreign_net_min_co_buy = int(thresholds.get("foreign_net_min_co_buy", 0) or 0)

    strat_volume_momentum = 1 if (float(sig.get("momentum_score") or 0.0) > momentum_score_high and yto > turnover_min) else 0
    strat_price_volume_new_high = 1 if (
        int(sig.get("is_price_breakout_400d", 0) or 0) == 1 and
        yto > turnover_min and
        int(sig.get("is_volume_breakout_10d", 0) or 0) == 1
    ) else 0
    strat_breakout_edge = 1 if (int(sig.get("is_near_40d_high", 0) or 0) == 1 and yto > turnover_min) else 0
    strat_trust_breakout = 1 if (int(sig.get("is_near_40d_high", 0) or 0) == 1 and yto > turnover_min and t3 > 0) else 0
    # 策略3（PDF）：投信動能連買（維持既有測試/定義）
    # 條件：動能>7.5 + 昨日成交金額>5e8 + 投信連買>=2
    try:
        ms = float(sig.get("momentum_score") or 0.0)
    except Exception:
        ms = 0.0
    try:
        trust_buy_streak = int(sig.get("trust_buy_streak") or 0)
    except Exception:
        trust_buy_streak = 0
    strat_trust_momentum_buy = 1 if (ms > momentum_score_high and yto > turnover_min and trust_buy_streak >= trust_buy_streak_min) else 0

    # 策略4（PDF）：外資剛大買（維持既有測試/定義）
    # 條件：動能>7.5 + 昨日成交金額>5e8 + 外資剛買第一天
    strat_foreign_big_buy = 1 if (ms > momentum_score_high and yto > turnover_min and int(sig.get("is_foreign_first_buy", 0) or 0) == 1) else 0

    # 策略5（PDF）：外資投信同買（維持既有測試/定義）
    # 條件：動能>6 + 昨日成交金額>5e8 + 土洋同買 + 外資買超>5千萬
    try:
        foreign_net = int(sig.get("foreign_net") or 0)
    except Exception:
        foreign_net = 0
    strat_co_buy = 1 if (ms > momentum_score_medium and yto > turnover_min and int(sig.get("is_co_buy", 0) or 0) == 1 and foreign_net > foreign_net_min_co_buy) else 0

    strat_volume_momentum_weak = 1 if (int(sig.get("is_volume_breakout_20d", 0) or 0) == 1 and int(sig.get("above_ma20", 0) or 0) == 0) else 0
    strat_price_volume_new_low = 1 if (int(sig.get("is_price_breakdown_20d", 0) or 0) == 1 and int(sig.get("is_volume_breakout_20d", 0) or 0) == 1) else 0
    strat_trust_breakdown = 1 if (int(sig.get("is_trust_first_sell", 0) or 0) == 1 and int(sig.get("is_price_breakdown_20d", 0) or 0) == 1) else 0
    strat_trust_momentum_sell = 1 if (int(sig.get("is_trust_sell_5d", 0) or 0) == 1 and (close is not None and ma20 is not None and close < ma20)) else 0
    strat_foreign_big_sell = 1 if (int(sig.get("is_foreign_first_sell", 0) or 0) == 1 or int(sig.get("is_foreign_sell_3d", 0) or 0) == 1) else 0
    strat_co_sell = 1 if int(sig.get("is_co_sell", 0) or 0) == 1 else 0

    # 策略分數（可解釋加分）
    if strat_volume_momentum:
        w = int(strat_points.get("strat_volume_momentum", 0) or 0)
        score_long += w; detail_long["strat_volume_momentum"] = w; rationale.add("volume_momentum")
    if strat_price_volume_new_high:
        w = int(strat_points.get("strat_price_volume_new_high", 0) or 0)
        score_long += w; detail_long["strat_price_volume_new_high"] = w; rationale.add("price_volume_new_high")
    if strat_breakout_edge:
        w = int(strat_points.get("strat_breakout_edge", 0) or 0)
        score_long += w; detail_long["strat_breakout_edge"] = w; rationale.add("breakout_edge")
    if strat_trust_breakout:
        w = int(strat_points.get("strat_trust_breakout", 0) or 0)
        score_long += w; detail_long["strat_trust_breakout"] = w; rationale.add("trust_breakout")
    if strat_trust_momentum_buy:
        w = int(strat_points.get("strat_trust_momentum_buy", 0) or 0)
        score_long += w; detail_long["strat_trust_momentum_buy"] = w; rationale.add("trust_momentum_buy")
    if strat_foreign_big_buy:
        w = int(strat_points.get("strat_foreign_big_buy", 0) or 0)
        score_long += w; detail_long["strat_foreign_big_buy"] = w; rationale.add("foreign_big_buy")
    if strat_co_buy:
        w = int(strat_points.get("strat_co_buy", 0) or 0)
        score_long += w; detail_long["strat_co_buy"] = w; rationale.add("co_buy")

    if strat_volume_momentum_weak:
        w = int(strat_points.get("strat_volume_momentum_weak", 0) or 0)
        score_short += w; detail_short["strat_volume_momentum_weak"] = w; rationale.add("volume_momentum_weak")
    if strat_price_volume_new_low:
        w = int(strat_points.get("strat_price_volume_new_low", 0) or 0)
        score_short += w; detail_short["strat_price_volume_new_low"] = w; rationale.add("price_volume_new_low")
    if strat_trust_breakdown:
        w = int(strat_points.get("strat_trust_breakdown", 0) or 0)
        score_short += w; detail_short["strat_trust_breakdown"] = w; rationale.add("trust_breakdown")
    if strat_trust_momentum_sell:
        w = int(strat_points.get("strat_trust_momentum_sell", 0) or 0)
        score_short += w; detail_short["strat_trust_momentum_sell"] = w; rationale.add("trust_momentum_sell")
    if strat_foreign_big_sell:
        w = int(strat_points.get("strat_foreign_big_sell", 0) or 0)
        score_short += w; detail_short["strat_foreign_big_sell"] = w; rationale.add("foreign_big_sell")
    if strat_co_sell:
        w = int(strat_points.get("strat_co_sell", 0) or 0)
        score_short += w; detail_short["strat_co_sell"] = w; rationale.add("co_sell")

    # 市場狀態權重（分 side）
    if market_regime in regime_weights:
        w_long = float(regime_weights.get(market_regime, {}).get("long", 1.0))
        w_short = float(regime_weights.get(market_regime, {}).get("short", 1.0))
        score_long = int(score_long * w_long)
        score_short = int(score_short * w_short)
        detail_long["regime_weight"] = f"{market_regime}_x{w_long:.2f}"
        detail_short["regime_weight"] = f"{market_regime}_x{w_short:.2f}"

    # --------- 步驟5：停損點（用 ATR 與 20日前高/前低）---------
    stop_side = None
    stop_price = None
    stop_pct = None

    atr = sig.get("atr14")
    # 需要 prev20 高低：用 MA / ATR 做保底
    if close is not None and atr is not None:
        if score_long >= score_short:
            stop_side = "long"
            stop_price = float(close) - 2.0 * float(atr)
        else:
            stop_side = "short"
            stop_price = float(close) + 2.0 * float(atr)

    if close is not None and stop_price is not None and close != 0:
        stop_pct = abs((float(close) - float(stop_price)) / float(close))

    sig.update({
        "strat_volume_momentum": strat_volume_momentum,
        "strat_price_volume_new_high": strat_price_volume_new_high,
        "strat_breakout_edge": strat_breakout_edge,
        "strat_trust_breakout": strat_trust_breakout,
        "strat_trust_momentum_buy": strat_trust_momentum_buy,
        "strat_foreign_big_buy": strat_foreign_big_buy,
        "strat_co_buy": strat_co_buy,
        "strat_volume_momentum_weak": strat_volume_momentum_weak,
        "strat_price_volume_new_low": strat_price_volume_new_low,
        "strat_trust_breakdown": strat_trust_breakdown,
        "strat_trust_momentum_sell": strat_trust_momentum_sell,
        "strat_foreign_big_sell": strat_foreign_big_sell,
        "strat_co_sell": strat_co_sell,
        "stop_loss_side": stop_side,
        "stop_loss_price": stop_price,
        "stop_loss_pct": stop_pct,
        "score_long": int(score_long),
        "score_short": int(score_short),
        "score_detail_long": detail_long,
        "score_detail_short": detail_short,
        "rationale_tags": sorted(list(rationale)),
    })

    # legacy score（暫時保留舊回測/工具）：用 dominant side 做正負
    has_long = any([strat_volume_momentum, strat_price_volume_new_high, strat_breakout_edge, strat_trust_breakout, strat_trust_momentum_buy, strat_foreign_big_buy, strat_co_buy])
    has_short = any([strat_volume_momentum_weak, strat_price_volume_new_low, strat_trust_breakdown, strat_trust_momentum_sell, strat_foreign_big_sell, strat_co_sell])
    if has_short and (score_short > score_long):
        sig["score"] = -int(score_short)
        sig["score_detail"] = detail_short
    else:
        sig["score"] = int(score_long)
        sig["score_detail"] = detail_long

    # ============================================================
    # 進出場規則（依動能教學 5 步驟可落地版本）
    # ============================================================
    # 步驟1：天氣圖閘門
    allow_long = market_regime in allowed_regimes_long
    allow_short = market_regime in allowed_regimes_short

    # range：保守一點，提高門檻
    min_long = int(entry_min_score_long) + (entry_min_score_range_boost if market_regime == "range" else 0)
    min_short = int(entry_min_score_short) + (entry_min_score_range_boost if market_regime == "range" else 0)

    long_strats = [
        ("volume_momentum", sig["strat_volume_momentum"]),
        ("price_volume_new_high", sig["strat_price_volume_new_high"]),
        ("trust_breakout", sig["strat_trust_breakout"]),
        ("trust_momentum_buy", sig["strat_trust_momentum_buy"]),
        ("foreign_big_buy", sig["strat_foreign_big_buy"]),
        ("co_buy", sig["strat_co_buy"]),
    ]
    short_strats = [
        ("volume_momentum_weak", sig.get("strat_volume_momentum_weak", 0)),
        ("price_volume_new_low", sig.get("strat_price_volume_new_low", 0)),
        ("trust_breakdown", sig.get("strat_trust_breakdown", 0)),
        ("trust_momentum_sell", sig.get("strat_trust_momentum_sell", 0)),
        ("foreign_big_sell", sig.get("strat_foreign_big_sell", 0)),
        ("co_sell", sig.get("strat_co_sell", 0)),
    ]

    primary = None
    for name, ok in long_strats:
        if ok == 1:
            primary = f"long:{name}"
            break
    if primary is None:
        for name, ok in short_strats:
            if ok == 1:
                primary = f"short:{name}"
                break

    # 步驟2+3+4：策略上榜 + 技術/法人確認（用已算出的旗標）
    entry_long = 1 if (allow_long and int(sig.get("score_long", 0) or 0) >= min_long and any(ok == 1 for _, ok in long_strats)) else 0
    entry_short = 1 if (allow_short and int(sig.get("score_short", 0) or 0) >= min_short and any(ok == 1 for _, ok in short_strats)) else 0

    # 步驟5：出場（進場理由消失 / 技術轉弱 / 法人轉賣）
    exit_long = 1 if (
        (sig.get("close") is not None and sig.get("ma20") is not None and sig["close"] < sig["ma20"]) or
        sig.get("is_price_breakdown_20d", 0) == 1 or
        sig.get("is_co_sell", 0) == 1 or
        sig.get("is_trust_sell_5d", 0) == 1
    ) else 0

    exit_short = 1 if (
        (sig.get("close") is not None and sig.get("ma20") is not None and sig["close"] > sig["ma20"]) or
        sig.get("is_price_breakout_20d", 0) == 1 or
        sig.get("is_co_buy", 0) == 1 or
        sig.get("is_trust_buy_5d", 0) == 1
    ) else 0

    sig.update({
        "primary_strategy": primary,
        "entry_long": entry_long,
        "entry_short": entry_short,
        "exit_long": exit_long,
        "exit_short": exit_short,
    })
    return sig


# ============================================================
# 8. 主流程：合併所有表 → 寫入 stock_signals_v2 + 印榜單
# ============================================================

def build_signals_for_date(target_date, *, strategy_cfg: dict, config_hash: str, config_snapshot: str):
    signals_rows = []
    with db_conn(commit_on_success=True) as conn:
        require_signals_table(conn)
        cursor = conn.cursor()
        cols = _get_table_columns(conn, "stock_signals_v2")

        lookback_days = int(strategy_cfg.get("signals", {}).get("lookback_days", getattr(settings, "SIGNALS_LOOKBACK_DAYS", 120)))
        lookback_start = target_date - timedelta(days=lookback_days)

        # 讀取日K（一次抓足夠的回溯，避免每檔股票 N+1 query）
        cursor.execute(
            """
            SELECT stock_id, trading_date, open, high, low, close, volume, turnover
            FROM stock_daily
            WHERE trading_date BETWEEN %s AND %s
            ORDER BY stock_id, trading_date
            """,
            (lookback_start, target_date),
        )
        daily_rows = cursor.fetchall()
        daily_by_stock = {}
        for r in daily_rows:
            daily_by_stock.setdefault(r["stock_id"], []).append(r)  # 舊→新

        # 法人（回溯 10 天用於連買/轉向）
        cursor.execute(
            """
            SELECT stock_id, trading_date, foreign_net, trust_net, dealer_net
            FROM institution_trading
            WHERE trading_date BETWEEN %s AND %s
            ORDER BY stock_id, trading_date DESC
            """,
            (target_date - timedelta(days=20), target_date),
        )
        inst_rows = cursor.fetchall()
        inst_by_stock = {}
        for r in inst_rows:
            inst_by_stock.setdefault(r["stock_id"], []).append(r)  # 新→舊（因為 ORDER BY DESC）

        # 融資（回溯 10 天）
        cursor.execute(
            """
            SELECT stock_id, trading_date, margin_balance
            FROM margin_trading
            WHERE trading_date BETWEEN %s AND %s
            ORDER BY stock_id, trading_date DESC
            """,
            (target_date - timedelta(days=20), target_date),
        )
        margin_rows = cursor.fetchall()
        margin_by_stock = {}
        for r in margin_rows:
            margin_by_stock.setdefault(r["stock_id"], []).append(r)  # 新→舊

        # 分點（當日）
        cursor.execute(
            """
            SELECT stock_id, trading_date, broker, branch, buy, sell, net
            FROM branch_trading
            WHERE trading_date=%s
            """,
            (target_date,),
        )
        branch_rows = cursor.fetchall()
        branch_by_stock = {}
        for r in branch_rows:
            branch_by_stock.setdefault(r["stock_id"], []).append(r)

        # 取得大盤 proxy（預設 0050，可在 settings.MARKET_PROXY_STOCK_ID 調整）
        proxy_id = strategy_cfg.get("signals", {}).get("market_proxy_stock_id", getattr(settings, "MARKET_PROXY_STOCK_ID", "0050"))
        proxy_lookback_days = int(strategy_cfg.get("signals", {}).get("proxy_lookback_days", getattr(settings, "SIGNALS_PROXY_LOOKBACK_DAYS", 260)))
        proxy_start = target_date - timedelta(days=proxy_lookback_days)
        cursor.execute(
            """
            SELECT trading_date, close
            FROM stock_daily
            WHERE stock_id=%s AND trading_date BETWEEN %s AND %s
            ORDER BY trading_date
            """,
            (proxy_id, proxy_start, target_date),
        )
        proxy_daily_asc = cursor.fetchall()  # 舊→新
        proxy_daily_desc = list(reversed(proxy_daily_asc))  # 新→舊
        market_regime = calc_market_regime(proxy_daily_desc)

        # === 每檔股票建 signals（只處理 target_date 有日K的股票） ===
        for stock_id, rows_asc in daily_by_stock.items():
            if not rows_asc:
                continue
            if rows_asc[-1]["trading_date"] != target_date:
                continue

            daily_desc = list(reversed(rows_asc))  # 新→舊
            sig = {"stock_id": stock_id, "trading_date": target_date, "market_regime": market_regime}

            # 400日新高/10日量新高需要更長回溯
            sig.update(calc_kbar_signals(daily_desc[:450]))
            sig.update(calc_institution_signals(inst_by_stock.get(stock_id, [])[:10]))
            sig.update(calc_margin_signals(margin_by_stock.get(stock_id, [])[:10]))
            branch_ratio = float(strategy_cfg.get("scoring", {}).get("thresholds", {}).get("branch_concentration_ratio", 0.0) or 0.0)
            sig.update(calc_branch_signals(branch_by_stock.get(stock_id, []), concentration_ratio=branch_ratio))

            # turnover / yesterday_turnover（以「上一筆 trading_date」當作前一交易日，天然處理週末/連假）
            today_row = rows_asc[-1]  # target_date
            y_row = rows_asc[-2] if len(rows_asc) >= 2 else None
            sig["turnover"] = today_row.get("turnover")
            sig["yesterday_turnover"] = (y_row.get("turnover") if y_row else None)

            # momentum_score（可落地版，數值約落在 0~20；缺值不爆炸）
            sig["momentum_score"] = calc_momentum_score(
                close=sig.get("close"),
                ma20=sig.get("ma20"),
                atr14=sig.get("atr14"),
                is_price_breakout_20d=int(sig.get("is_price_breakout_20d", 0) or 0),
            )

            sig = calc_strategies_and_score(market_regime, sig, strategy_cfg)
            sig["config_hash"] = config_hash
            sig["config_snapshot"] = config_snapshot
            sig["signal_version"] = strategy_cfg.get("signals", {}).get("signal_version", "v2")

            # JSON 欄位：pymysql 需轉成 str（DB 沒該欄位也不影響 insert，因為我們會動態決定欄位清單）
            if isinstance(sig.get("score_detail"), (dict, list)):
                sig["score_detail"] = json.dumps(sig["score_detail"], ensure_ascii=False)
            if isinstance(sig.get("score_detail_long"), (dict, list)):
                sig["score_detail_long"] = json.dumps(sig["score_detail_long"], ensure_ascii=False)
            if isinstance(sig.get("score_detail_short"), (dict, list)):
                sig["score_detail_short"] = json.dumps(sig["score_detail_short"], ensure_ascii=False)
            if isinstance(sig.get("rationale_tags"), (dict, list)):
                sig["rationale_tags"] = json.dumps(sig["rationale_tags"], ensure_ascii=False)

            signals_rows.append(sig)

        # 批次寫入
        if signals_rows:
            base_cols = [
                "stock_id", "trading_date", "market_regime",
                "score", "score_detail",
                "close", "volume", "turnover", "yesterday_turnover", "momentum_score",
                "ma20", "ma60", "atr14", "above_ma20", "above_ma60",
                "is_price_breakout_20d", "is_price_breakdown_20d", "is_volume_breakout_20d",
                "dist_40d_high_pct", "is_near_40d_high", "is_price_breakout_400d", "is_volume_breakout_10d",
                "foreign_net", "trust_net", "dealer_net", "foreign_net_3d", "trust_net_3d", "trust_net_5d",
                "is_foreign_first_buy", "is_trust_first_buy", "is_co_buy", "is_foreign_buy_3d", "is_trust_buy_3d", "is_trust_buy_5d",
                "is_foreign_first_sell", "is_trust_first_sell", "is_foreign_sell_3d", "is_trust_sell_5d", "is_co_sell",
                "margin_balance", "margin_balance_chg_3d", "is_margin_risk",
                "branch_top5_buy_ratio", "branch_top5_net_ratio", "is_branch_concentration",
                "strat_volume_momentum", "strat_price_volume_new_high", "strat_breakout_edge", "strat_trust_breakout", "strat_trust_momentum_buy",
                "strat_foreign_big_buy", "strat_co_buy",
                "strat_volume_momentum_weak", "strat_price_volume_new_low", "strat_trust_breakdown", "strat_trust_momentum_sell",
                "strat_foreign_big_sell", "strat_co_sell",
                "stop_loss_side", "stop_loss_price", "stop_loss_pct",
                "primary_strategy", "entry_long", "exit_long", "entry_short", "exit_short",
            ]
            optional_cols = [
                "score_long",
                "score_short",
                "score_detail_long",
                "score_detail_short",
                "rationale_tags",
                "config_hash",
                "config_snapshot",
                "signal_version",
            ]
            cols_to_write = [c for c in base_cols if c in cols] + [c for c in optional_cols if c in cols]
            cols_sql = ", ".join(cols_to_write)
            vals_sql = ", ".join([f"%({c})s" for c in cols_to_write])
            insert_sql = f"REPLACE INTO stock_signals_v2 ({cols_sql}) VALUES ({vals_sql})"
            cursor.executemany(insert_sql, signals_rows)

        cursor.close()

    # 印出榜單（不依賴 DB）
    top_n = int(strategy_cfg.get("signals", {}).get("top_n", getattr(settings, "SIGNALS_TOP_N", 30)))
    longs = sorted(signals_rows, key=lambda x: int(x.get("score_long", 0) or 0), reverse=True)[:top_n]
    shorts = sorted(signals_rows, key=lambda x: int(x.get("score_short", 0) or 0), reverse=True)[:top_n]

    def _fmt_row(r):
        tags = []
        for k in (
            "strat_volume_momentum",
            "strat_price_volume_new_high",
            "strat_breakout_edge",
            "strat_trust_breakout",
            "strat_trust_momentum_buy",
            "strat_foreign_big_buy",
            "strat_co_buy",
        ):
            if r.get(k) == 1:
                tags.append(k.replace("strat_", ""))
        return f"{r['stock_id']} scoreL={int(r.get('score_long',0) or 0)} scoreS={int(r.get('score_short',0) or 0)} tags={','.join(tags) or '-'}"

    print(f"[OK] signals built for {target_date} | market_regime={market_regime} | count={len(signals_rows)}")
    print("=== 強勢 Top ===")
    for r in longs[: min(10, len(longs))]:
        print(_fmt_row(r))
    print("=== 弱勢 Top ===")
    for r in shorts[: min(10, len(shorts))]:
        print(_fmt_row(r))


if __name__ == "__main__":
    # 支援：
    # - python etl/build_signals.py                  （自動挑最新完整交易日）
    # - python etl/build_signals.py YYYY-MM-DD       （單日）
    # - python etl/build_signals.py START END        （區間，含 start/end）
    args = list(sys.argv[1:])
    overrides = None
    if "--override" in args:
        idx = args.index("--override")
        if idx + 1 < len(args):
            try:
                overrides = json.loads(args[idx + 1])
            except Exception as e:
                raise RuntimeError(f"無法解析 --override JSON：{e}")
            del args[idx: idx + 2]
    arg_date = None
    arg_range = None
    if len(args) >= 2:
        arg_range = (datetime.strptime(args[0], "%Y-%m-%d").date(), datetime.strptime(args[1], "%Y-%m-%d").date())
    elif len(args) == 1:
        arg_date = datetime.strptime(args[0], "%Y-%m-%d").date()

    strategy_cfg, config_hash, config_snapshot = load_strategy_config(overrides)

    with db_conn() as conn:
        with conn.cursor() as cur:
            # 若未指定日期，挑「最新且資料量足夠」的交易日（避免像 2025-12-12 只有 263 檔的半套資料）
            min_stocks = int(strategy_cfg.get("signals", {}).get("min_stocks", getattr(settings, "SIGNALS_MIN_STOCKS", 2000)))
            if arg_date is None:
                cur.execute(
                    """
                    SELECT trading_date, COUNT(DISTINCT stock_id) AS stocks
                    FROM stock_daily
                    GROUP BY trading_date
                    ORDER BY trading_date DESC
                    LIMIT 30
                    """
                )
                rows = cur.fetchall()
                chosen = None
                for r in rows:
                    if int(r["stocks"]) >= int(min_stocks):
                        chosen = r["trading_date"]
                        break
                if chosen is None:
                    # fallback：就用最大日期（即使不完整）
                    cur.execute("SELECT MAX(trading_date) AS maxd FROM stock_daily")
                    chosen = cur.fetchone()["maxd"]
                arg_date = chosen

    if arg_range:
        start, end = arg_range
        with db_conn() as cal_conn:
            with cal_conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT trading_date
                    FROM stock_daily
                    WHERE trading_date BETWEEN %s AND %s
                    ORDER BY trading_date
                    """,
                    (start, end),
                )
                days = [r["trading_date"] for r in cur.fetchall()]

        for d in days:
            build_signals_for_date(d, strategy_cfg=strategy_cfg, config_hash=config_hash, config_snapshot=config_snapshot)
    else:
        build_signals_for_date(arg_date, strategy_cfg=strategy_cfg, config_hash=config_hash, config_snapshot=config_snapshot)

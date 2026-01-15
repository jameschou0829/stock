from datetime import date, timedelta
import pandas as pd

from utils.db import db_conn
from utils.fetcher import FinMindPaymentRequiredError, FinMindAuthError, finmind_get_data
from utils.fetch_log import get_last_date as _get_last_date, upsert_fetch_log as _upsert_fetch_log

# 預設抓到今天；若「今天資料還沒出」，上層可傳入 effective_end_date 讓 ETL 先補到最近可用交易日
TARGET_END_DATE = date.today()
DEFAULT_START_DATE = date(2015, 1, 1)
TABLE_NAME = "stock_daily"

def _find_column_case_insensitive(df: pd.DataFrame, want_lower: str):
    """
    FinMind 回傳欄位偶爾會有大小寫差異；此 helper 用來做大小寫兼容。
    例：Trading_money / Trading_Money / trading_money
    """
    mapping = {str(c).lower(): c for c in df.columns}
    return mapping.get(want_lower)


def _find_first_column(df: pd.DataFrame, want_lowers):
    """
    want_lowers: List[str]（全部用 lower case）
    依序找第一個存在的欄位名（大小寫兼容）。
    """
    for w in want_lowers:
        col = _find_column_case_insensitive(df, w)
        if col:
            return col
    return None


def _safe_int(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return int(v)
    except Exception:
        return None


def _safe_float(v):
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        return float(v)
    except Exception:
        return None



def fetch_from_finmind(stock_id, start_date, end_date):
    params = {
        "data_id": stock_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }
    data = finmind_get_data("TaiwanStockPrice", params, timeout=20, max_retry=3, wait_seconds=2)
    return pd.DataFrame(data)


def save_daily_batch(df: pd.DataFrame):
    if df.empty:
        return

    rows = []

    # FinMind TaiwanStockPrice 欄位（可能有大小寫差異/命名差異）
    stock_col = _find_first_column(df, ["stock_id", "data_id"])
    date_col = _find_first_column(df, ["date"])
    open_col = _find_first_column(df, ["open"])
    high_col = _find_first_column(df, ["high", "max"])  # FinMind 常用 max 表示 high
    low_col = _find_first_column(df, ["low", "min"])    # FinMind 常用 min 表示 low
    close_col = _find_first_column(df, ["close"])
    vol_col = _find_first_column(df, ["trading_volume", "volume"])
    money_col = _find_first_column(df, ["trading_money"])  # Trading_money / Trading_Money / trading_money

    for _, r in df.iterrows():
        stock_id = r.get(stock_col) if stock_col else None
        d = r.get(date_col) if date_col else None
        if stock_id is None or d is None:
            # 無主鍵就跳過，避免寫入炸裂
            continue

        open_v = _safe_float(r.get(open_col)) if open_col else None
        high_v = _safe_float(r.get(high_col)) if high_col else None
        low_v = _safe_float(r.get(low_col)) if low_col else None
        close = _safe_float(r.get(close_col)) if close_col else None
        volume = _safe_int(r.get(vol_col)) if vol_col else None

        # 成交金額 turnover（TWD）
        # 1) 優先使用 FinMind 回傳的 Trading_money（大小寫兼容）
        # 2) 若資料源缺 Trading_money，fallback 用 close * Trading_Volume 推估（僅供教學/Debug）
        turnover = None
        if money_col:
            turnover = _safe_int(r.get(money_col))
        if turnover is None and close is not None and volume is not None:
            # fallback：以收盤價 * 成交股數估算成交金額（非官方成交金額；可先讓策略跑通）
            turnover = int(round(close * float(volume)))

        rows.append((
            stock_id,
            d,                   # → trading_date
            open_v,
            high_v,              # → high
            low_v,               # → low
            close,
            volume,              # → volume（允許 NULL）
            turnover,            # → turnover（成交金額）
        ))

    with db_conn(commit_on_success=True) as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO stock_daily
                (stock_id, trading_date, open, high, low, close, volume, turnover)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                ON DUPLICATE KEY UPDATE
                  open=VALUES(open),
                  high=VALUES(high),
                  low=VALUES(low),
                  close=VALUES(close),
                  volume=VALUES(volume),
                  turnover=VALUES(turnover)
                """,
                rows,
            )


def run_daily(stock_id: str, end_date: date = TARGET_END_DATE):
    try:
        # checkpoint：以 fetch_log 為主
        last_date = _get_last_date(TABLE_NAME, stock_id)

        if last_date is None:
            start_date = DEFAULT_START_DATE
        else:
            start_date = last_date + timedelta(days=1)

        if start_date > end_date:
            print(f"[SKIP] {stock_id} already complete")
            return

        df = fetch_from_finmind(stock_id, start_date, end_date)

        if df.empty:
            print(f"[WARN] {stock_id} no new data")
            return

        save_daily_batch(df)

        # 更新 fetch_log（取本次資料最大日期）
        try:
            max_s = str(df[_find_first_column(df, ["date"])].max())
            max_d = date(int(max_s[0:4]), int(max_s[5:7]), int(max_s[8:10]))
            _upsert_fetch_log(TABLE_NAME, stock_id, max_d)
        except Exception:
            pass

        print(
            f"[OK] {stock_id} daily updated "
            f"{start_date} → {df['date'].max()}"
        )

    except (FinMindPaymentRequiredError, FinMindAuthError):
        # 讓上層（run_daily_etl.py）可以整批中止，避免刷一堆 error
        raise
    except Exception as e:
        print(f"[ERROR] {stock_id} 日K 失敗: {e}")

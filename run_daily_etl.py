# run_daily_etl.py
from etl.fetch_daily import run_daily
from etl.fetch_institution import run_institution
from etl.fetch_margin import run_margin
from etl.fetch_branch import run_branch
from utils.db import db_conn
from utils.fetcher import FinMindPaymentRequiredError, FinMindAuthError, fetch_with_retry, finmind_get_data
from datetime import date, timedelta
from configs import settings
from utils.checkpoint import load_checkpoint, save_checkpoint, today_str
from utils.fail_log import log_fail
import os
from typing import Optional


def get_active_stock_list():
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT stock_id FROM stock_info WHERE is_active = 1")
            rows = cur.fetchall()
            return [r["stock_id"] for r in rows]


def run_daily_all():
    stock_list = get_active_stock_list()

    print(f"[INFO] 今日需處理股票數：{len(stock_list)}")

    # checkpoint：避免跑到一半中斷後，下次又從頭打 API
    ckpt_path = os.path.join(os.path.dirname(__file__), ".etl_checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)

    def _pick_effective_end_date() -> Optional[date]:
        """
        若 settings.ETL_REQUIRE_TODAY_PRICE_DATA=True，且 FinMind 今天資料尚未出，
        就改用「最近可用交易日」當 end_date，讓你可以補昨天/前一交易日的缺口。
        """
        check_sid = getattr(settings, "ETL_TODAY_CHECK_STOCK_ID", "2330")
        today_s = date.today().strftime("%Y-%m-%d")
        # 先試今天
        rows = finmind_get_data(
            "TaiwanStockPrice",
            {"data_id": check_sid, "start_date": today_s, "end_date": today_s},
            timeout=20,
            max_retry=1,
            wait_seconds=0,
        )
        has_today = any((r.get("date") == today_s) for r in (rows or []))
        if has_today:
            return date.today()

        # 今天沒資料：從近 10 天找 latest date（避免整批直接 return）
        lookback_start = (date.today() - timedelta(days=10)).strftime("%Y-%m-%d")
        rows2 = finmind_get_data(
            "TaiwanStockPrice",
            {"data_id": check_sid, "start_date": lookback_start, "end_date": today_s},
            timeout=20,
            max_retry=1,
            wait_seconds=0,
        )
        ds = [r.get("date") for r in (rows2 or []) if r.get("date")]
        if not ds:
            return None
        latest_s = max(ds)
        try:
            y = int(latest_s[0:4]); m = int(latest_s[5:7]); d = int(latest_s[8:10])
            return date(y, m, d)
        except Exception:
            return None

    effective_end_date = date.today()
    if getattr(settings, "ETL_REQUIRE_TODAY_PRICE_DATA", True):
        picked = _pick_effective_end_date()
        if not picked:
            print("[SKIP] 無法取得最新可用交易日（FinMind 近 10 天無回應），整批 ETL 不跑。")
            return
        if picked != date.today():
            print(f"[WARN] FinMind 尚未提供今日日K資料（{date.today()}），改用最近可用交易日 {picked} 補跑。")
        effective_end_date = picked

    ckpt_as_of = effective_end_date.strftime("%Y-%m-%d")
    if ckpt.get("as_of") == ckpt_as_of and ckpt.get("last_stock_id") in stock_list:
        last_id = ckpt["last_stock_id"]
        idx = stock_list.index(last_id) + 1
        if idx < len(stock_list):
            print(f"[RESUME] 從 checkpoint 繼續：上次到 {last_id}，本次從 index={idx} 開始")
            stock_list = stock_list[idx:]

    # 開跑前先做一次健康檢查，避免 token/方案問題刷出一堆 error
    try:
        # 仍保留 token/方案可用性測試（避免剛好 check_sid 當天停牌造成誤判）
        # 另外做 token/方案可用性測試（避免剛好 check_sid 當天停牌造成誤判）
        proxy_id = getattr(settings, "MARKET_PROXY_STOCK_ID", "0050")
        params = {
            "data_id": proxy_id,
            "start_date": (date.today() - timedelta(days=5)).strftime("%Y-%m-%d"),
            "end_date": date.today().strftime("%Y-%m-%d"),
        }
        fetch_with_retry("TaiwanStockPrice", params, max_retry=1, wait_seconds=0)
    except (FinMindPaymentRequiredError, FinMindAuthError) as e:
        print(f"[FATAL] FinMind token/方案錯誤：{e}")
        print("建議：更新 FINMIND_API_TOKEN（可用環境變數 export FINMIND_API_TOKEN=...）後再重跑。")
        return

    for sid in stock_list:
        try:
            try:
                run_daily(sid, end_date=effective_end_date)
            except Exception as e:
                log_fail(table_name="stock_daily", stock_id=sid, stage="run_daily", error=e)
                raise
            try:
                run_institution(sid, end_date=effective_end_date)
            except Exception as e:
                log_fail(table_name="institution_trading", stock_id=sid, stage="run_institution", error=e)
                raise
            try:
                run_margin(sid, end_date=effective_end_date)
            except Exception as e:
                log_fail(table_name="margin_trading", stock_id=sid, stage="run_margin", error=e)
                raise
            try:
                run_branch(sid)
            except Exception as e:
                log_fail(table_name="branch_trading", stock_id=sid, stage="run_branch", error=e)
                raise
            save_checkpoint(ckpt_path, {"as_of": ckpt_as_of, "last_stock_id": sid})
        except (FinMindPaymentRequiredError, FinMindAuthError) as e:
            print(f"[FATAL] FinMind token/方案錯誤（中止整批）：{e}")
            break
        except Exception as e:
            # 非 token 類錯誤：記錄後繼續下一檔（可續跑）
            print(f"[ERROR] {sid} ETL 失敗（已記錄 fail_log）：{e}")
            save_checkpoint(ckpt_path, {"as_of": ckpt_as_of, "last_stock_id": sid})
            continue


if __name__ == "__main__":
    run_daily_all()

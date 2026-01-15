# configs/settings.py
import os
from pathlib import Path

# 自動載入專案根目錄的 .env（若存在）
# - 讓你不用手動 export，也能讓 run_daily_etl.py 讀到 FINMIND_API_TOKEN
# - 若系統環境變數已設定，預設不覆蓋（override=False）
try:
    from dotenv import load_dotenv  # type: ignore

    _ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(dotenv_path=_ROOT / ".env", override=False)
except Exception:
    # 沒裝 python-dotenv 或其他環境問題：就退回只讀 os.getenv
    pass


def require_finmind_token() -> str:
    """
    本專案禁止在程式碼內硬編 token，只允許從環境變數讀取。
    這裡提供一個「用到才檢查」的 helper，避免 web/backtest 不跑 ETL 時也被 token 綁死。
    """
    token = os.getenv("FINMIND_API_TOKEN")
    if not token:
        raise RuntimeError(
            "缺少環境變數 FINMIND_API_TOKEN。"
            "請先設定：export FINMIND_API_TOKEN='...'\n"
            "（參考 env.example / README）"
        )
    return token
FINMIND_API_URL = os.getenv("FINMIND_API_URL", "http://api.finmindtrade.com/api/v4/data")

DB_HOST = "127.0.0.1"
DB_PORT = 3307
DB_USER = "root"
DB_PASS = "root"
DB_NAME = "stock_test2"
DB_CHARSET = "utf8mb4"

# ==========================
# Signals / 動能選股設定
# ==========================
# 步驟1：大盤天氣圖 proxy（用來近似「偏多/震盪/偏空」）
# 預設用 0050（數字股，會在 stock_info/is_active 流程內存在）
MARKET_PROXY_STOCK_ID = "0050"

# build_signals.py 輸出榜單筆數
SIGNALS_TOP_N = 30

# 產訊號時抓個股日K回溯天數（用於 MA/突破/量突破）
# 由於多方策略「價量創新高」需要 400 日新高（約 1.5 年的交易日），這裡預設拉到 520 天。
SIGNALS_LOOKBACK_DAYS = 520

# 大盤天氣圖 proxy 回溯天數（要足夠算 MA60）
SIGNALS_PROXY_LOOKBACK_DAYS = 260

# 自動挑「最新且資料完整」交易日的門檻（distinct stock_id 數）
SIGNALS_MIN_STOCKS = 2000

# ==========================
# FinMind API 節流（避免超過 6000/hr）
# ==========================
# 6000/hr ≈ 0.6s/次；用 0.65 留點安全邊際
FINMIND_MIN_INTERVAL_SEC = float(os.getenv("FINMIND_MIN_INTERVAL_SEC", "0.65"))

# ==========================
# ETL 行為
# ==========================
# 若為 True：只要 API 還拿不到「今天」的日K，就整批不跑（避免打到半套日）
ETL_REQUIRE_TODAY_PRICE_DATA = os.getenv("ETL_REQUIRE_TODAY_PRICE_DATA", "1") == "1"

# 用哪一檔來判斷「今天日K是否已釋出」：建議用流動性高、幾乎每天都有資料的股票
ETL_TODAY_CHECK_STOCK_ID = os.getenv("ETL_TODAY_CHECK_STOCK_ID", "2330")

# ==========================
# 進出場參數（動能策略）
# ==========================
# 進場最低分數（多/空用正負）
SIGNALS_ENTRY_MIN_SCORE = int(os.getenv("SIGNALS_ENTRY_MIN_SCORE", "25"))

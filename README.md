## stock_test3（動能策略：ETL / Signals / Backtest / Web）

### 先決條件
- Python 3.10+
- MySQL（專案預設：`127.0.0.1:3307`）

### 1) 設定環境變數
本專案 **不允許** 在程式碼內硬編 FinMind token。請用環境變數提供。

參考 `.env.example`，你可以：

```bash
export FINMIND_API_TOKEN="YOUR_TOKEN"
export DB_HOST="127.0.0.1"
export DB_PORT="3307"
export DB_USER="root"
export DB_PASS="root"
export DB_NAME="stock_test2"
export DB_CHARSET="utf8mb4"
```

### 1.1) 策略設定（strategy.yaml）
所有策略門檻/係數統一集中在 `configs/strategy.yaml`，覆蓋優先序：
1) CLI / Web request body
2) 環境變數（`STRATEGY__...`）
3) YAML 預設值

範例（環境變數覆蓋）：
```bash
export STRATEGY__COSTS__COMMISSION_BPS=10
export STRATEGY__SCORING__ENTRY_MIN_SCORE_LONG=30
```

### 2) 安裝依賴

```bash
python -m pip install -r requirements.txt
```

### 3) 套用 migrations（DDL）
`etl/build_signals.py` 不會在 runtime 變更 schema；請先自行套用 `migrations/*.sql`。
新增機率欄位與回測 entry metadata 時，也需要套用：
- `migrations/20260115_add_prob_column.sql`
- `migrations/20260115_backtest_trades_entry_metadata.sql`
- `migrations/20260115_extend_stock_signals_v2.sql`

### 4) 跑全市場 ETL（每日增量）

```bash
python run_daily_etl.py
```

### 5) 產生 signals（單日或區間）

```bash
python etl/build_signals.py 2026-01-13
# 或
python etl/build_signals.py 2026-01-01 2026-01-13
```

### 6) 跑回測（CLI）

```bash
python backtest/backtest_signals.py --start 2025-09-11 --end 2026-01-09 --side long --top-n 20 --holding-days 0 --use-stop-loss --use-entry-exit-signals
```

### 6.1) 訓練上漲機率（LightGBM）
```bash
python tools/train_probability.py --start 2025-01-01 --end 2026-01-13
```

### 7) 啟動 Web
本專案前端為 **FastAPI + Jinja2 templates**，不需要 npm build。

```bash
uvicorn webapp.main:app --reload --host 127.0.0.1 --port 8000
```

- `http://127.0.0.1:8000/`：回測頁
- `http://127.0.0.1:8000/entries`：榜單頁（可排序、顯示上榜天數/乖離）

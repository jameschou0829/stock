-- Phase B：ETL 穩定性 / 可續跑 / 可觀測

-- 1) fail_log：記錄 ETL 失敗（不影響主流程可繼續跑）
CREATE TABLE IF NOT EXISTS fail_log (
  id BIGINT NOT NULL AUTO_INCREMENT,
  table_name VARCHAR(64) NOT NULL,
  stock_id VARCHAR(16) NOT NULL,
  stage VARCHAR(64) NOT NULL,
  trading_date VARCHAR(16) NULL,
  error_type VARCHAR(128) NOT NULL,
  error_message TEXT NOT NULL,
  traceback MEDIUMTEXT NULL,
  created_at DATETIME NOT NULL,
  PRIMARY KEY (id),
  INDEX idx_fail_log_stock (stock_id, created_at),
  INDEX idx_fail_log_table (table_name, created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 2) 索引建議：提高 upsert / 查詢效率
-- stock_daily / institution_trading / margin_trading / branch_trading 都建議有 (stock_id, trading_date) 索引
-- （若已是 PRIMARY KEY 可略過）
CREATE INDEX idx_stock_daily_sid_date ON stock_daily (stock_id, trading_date);
CREATE INDEX idx_inst_sid_date ON institution_trading (stock_id, trading_date);
CREATE INDEX idx_margin_sid_date ON margin_trading (stock_id, trading_date);
CREATE INDEX idx_branch_sid_date ON branch_trading (stock_id, trading_date);

-- 3) BIGINT 升級（避免買賣量/金額溢位）
ALTER TABLE institution_trading
  MODIFY foreign_buy BIGINT NULL,
  MODIFY foreign_sell BIGINT NULL,
  MODIFY foreign_net BIGINT NULL,
  MODIFY trust_buy BIGINT NULL,
  MODIFY trust_sell BIGINT NULL,
  MODIFY trust_net BIGINT NULL,
  MODIFY dealer_buy BIGINT NULL,
  MODIFY dealer_sell BIGINT NULL,
  MODIFY dealer_net BIGINT NULL;

ALTER TABLE branch_trading
  MODIFY buy BIGINT NULL,
  MODIFY sell BIGINT NULL,
  MODIFY net BIGINT NULL;

ALTER TABLE margin_trading
  MODIFY margin_buy BIGINT NULL,
  MODIFY margin_sell BIGINT NULL,
  MODIFY margin_balance BIGINT NULL,
  MODIFY short_buy BIGINT NULL,
  MODIFY short_sell BIGINT NULL,
  MODIFY short_balance BIGINT NULL;

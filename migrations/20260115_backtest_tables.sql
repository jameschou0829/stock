-- Backtest runs / trades / equity curve

CREATE TABLE IF NOT EXISTS backtest_runs (
  run_id VARCHAR(36) NOT NULL,
  start_date DATE NOT NULL,
  end_date DATE NOT NULL,
  side VARCHAR(8) NOT NULL,
  config_hash VARCHAR(64) NOT NULL,
  config_snapshot JSON NOT NULL,
  created_at DATETIME NOT NULL,
  PRIMARY KEY (run_id),
  INDEX idx_backtest_runs_date (start_date, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS backtest_trades (
  id BIGINT NOT NULL AUTO_INCREMENT,
  run_id VARCHAR(36) NOT NULL,
  stock_id VARCHAR(16) NOT NULL,
  side VARCHAR(8) NOT NULL,
  signal_date DATE NULL,
  entry_exec_date DATE NULL,
  entry_timing VARCHAR(16) NULL,
  entry_price DOUBLE NULL,
  entry_score DOUBLE NULL,
  entry_primary_strategy VARCHAR(64) NULL,
  entry_rationale_tags JSON NULL,
  entry_prob DOUBLE NULL,
  exit_date DATE NOT NULL,
  exit_price DOUBLE NULL,
  ret_gross DOUBLE NOT NULL,
  ret_net DOUBLE NOT NULL,
  cost_paid DOUBLE NOT NULL,
  exit_reason VARCHAR(32) NULL,
  kpi_passed TINYINT NOT NULL DEFAULT 0,
  PRIMARY KEY (id),
  INDEX idx_backtest_trades_run (run_id),
  INDEX idx_backtest_trades_stock_date (stock_id, entry_exec_date),
  INDEX idx_backtest_trades_signal_date (signal_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS backtest_equity_curve (
  id BIGINT NOT NULL AUTO_INCREMENT,
  run_id VARCHAR(36) NOT NULL,
  trading_date DATE NOT NULL,
  equity DOUBLE NOT NULL,
  drawdown DOUBLE NOT NULL,
  cash DOUBLE NOT NULL,
  positions_count INT NOT NULL,
  PRIMARY KEY (id),
  INDEX idx_backtest_equity_run_date (run_id, trading_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

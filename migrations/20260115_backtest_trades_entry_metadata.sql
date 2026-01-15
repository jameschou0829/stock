-- Add entry metadata columns for backtest_trades
ALTER TABLE backtest_trades
  ADD COLUMN signal_date DATE NULL,
  ADD COLUMN entry_exec_date DATE NULL,
  ADD COLUMN entry_timing VARCHAR(16) NULL,
  ADD COLUMN entry_price DOUBLE NULL,
  ADD COLUMN entry_primary_strategy VARCHAR(64) NULL,
  ADD COLUMN entry_rationale_tags JSON NULL,
  ADD COLUMN entry_score DOUBLE NULL,
  ADD COLUMN entry_prob DOUBLE NULL,
  ADD COLUMN exit_price DOUBLE NULL,
  ADD COLUMN cost_paid DOUBLE NULL;

CREATE INDEX idx_backtest_trades_signal_date ON backtest_trades (signal_date);

-- Add probability column for signals
ALTER TABLE stock_signals_v2
  ADD COLUMN prob_kpi_5d_2pct_touch DOUBLE NULL;

CREATE INDEX idx_signals_prob_kpi_5d_2pct_touch
  ON stock_signals_v2 (trading_date, prob_kpi_5d_2pct_touch);

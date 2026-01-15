-- Extend stock_signals_v2 with long/short scores, rationale, prob, config hash, signal version
ALTER TABLE stock_signals_v2
  ADD COLUMN score_long INT NULL,
  ADD COLUMN score_short INT NULL,
  ADD COLUMN score_detail_long JSON NULL,
  ADD COLUMN score_detail_short JSON NULL,
  ADD COLUMN rationale_tags JSON NULL,
  ADD COLUMN prob_kpi_5d_2pct_touch DOUBLE NULL,
  ADD COLUMN config_hash VARCHAR(64) NULL,
  ADD COLUMN signal_version VARCHAR(16) NULL;

CREATE INDEX idx_score_long ON stock_signals_v2 (trading_date, score_long);
CREATE INDEX idx_score_short ON stock_signals_v2 (trading_date, score_short);

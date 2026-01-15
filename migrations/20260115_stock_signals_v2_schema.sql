-- 定版：stock_signals_v2 schema（禁止 runtime ALTER/CREATE）
-- 用法：請先在 MySQL 內套用本檔，再執行 `python etl/build_signals.py ...`

CREATE TABLE IF NOT EXISTS stock_signals_v2 (
  stock_id VARCHAR(16) NOT NULL,
  trading_date DATE NOT NULL,

  -- Market regime
  market_regime VARCHAR(16) NOT NULL,

  -- Scores（v1: score；v2: score_long/score_short）
  score INT NOT NULL DEFAULT 0,
  score_long INT NULL,
  score_short INT NULL,
  score_detail JSON NULL,
  score_detail_long JSON NULL,
  score_detail_short JSON NULL,
  rationale_tags JSON NULL,
  config_hash VARCHAR(64) NULL,
  config_snapshot JSON NULL,
  prob_kpi_5d_2pct_touch DOUBLE NULL,

  -- K 棒/技術
  close DOUBLE NULL,
  volume BIGINT NULL,
  turnover BIGINT NULL,
  yesterday_turnover BIGINT NULL,
  momentum_score DOUBLE NULL,
  ma20 DOUBLE NULL,
  ma60 DOUBLE NULL,
  atr14 DOUBLE NULL,
  above_ma20 TINYINT NOT NULL DEFAULT 0,
  above_ma60 TINYINT NOT NULL DEFAULT 0,
  is_price_breakout_20d TINYINT NOT NULL DEFAULT 0,
  is_price_breakdown_20d TINYINT NOT NULL DEFAULT 0,
  is_volume_breakout_20d TINYINT NOT NULL DEFAULT 0,
  dist_40d_high_pct DOUBLE NULL,
  is_near_40d_high TINYINT NOT NULL DEFAULT 0,
  is_price_breakout_400d TINYINT NOT NULL DEFAULT 0,
  is_volume_breakout_10d TINYINT NOT NULL DEFAULT 0,

  -- 法人（籌碼）
  foreign_net BIGINT NULL,
  trust_net BIGINT NULL,
  dealer_net BIGINT NULL,
  foreign_net_3d BIGINT NULL,
  trust_net_3d BIGINT NULL,
  trust_net_5d BIGINT NULL,
  is_foreign_first_buy TINYINT NOT NULL DEFAULT 0,
  is_trust_first_buy TINYINT NOT NULL DEFAULT 0,
  is_co_buy TINYINT NOT NULL DEFAULT 0,
  is_foreign_buy_3d TINYINT NOT NULL DEFAULT 0,
  is_trust_buy_3d TINYINT NOT NULL DEFAULT 0,
  is_trust_buy_5d TINYINT NOT NULL DEFAULT 0,
  is_foreign_first_sell TINYINT NOT NULL DEFAULT 0,
  is_trust_first_sell TINYINT NOT NULL DEFAULT 0,
  is_foreign_sell_3d TINYINT NOT NULL DEFAULT 0,
  is_trust_sell_5d TINYINT NOT NULL DEFAULT 0,
  is_co_sell TINYINT NOT NULL DEFAULT 0,

  -- 融資融券（風險）
  margin_balance BIGINT NULL,
  margin_balance_chg_3d BIGINT NULL,
  is_margin_risk TINYINT NOT NULL DEFAULT 0,

  -- 分點（簡化集中度）
  branch_top5_buy_ratio DOUBLE NULL,
  branch_top5_net_ratio DOUBLE NULL,
  is_branch_concentration TINYINT NOT NULL DEFAULT 0,

  -- 策略標籤（步驟2）
  strat_volume_momentum TINYINT NOT NULL DEFAULT 0,
  strat_price_volume_new_high TINYINT NOT NULL DEFAULT 0,
  strat_breakout_edge TINYINT NOT NULL DEFAULT 0,
  strat_trust_breakout TINYINT NOT NULL DEFAULT 0,
  strat_trust_momentum_buy TINYINT NOT NULL DEFAULT 0,
  strat_foreign_big_buy TINYINT NOT NULL DEFAULT 0,
  strat_co_buy TINYINT NOT NULL DEFAULT 0,
  strat_volume_momentum_weak TINYINT NOT NULL DEFAULT 0,
  strat_price_volume_new_low TINYINT NOT NULL DEFAULT 0,
  strat_trust_breakdown TINYINT NOT NULL DEFAULT 0,
  strat_trust_momentum_sell TINYINT NOT NULL DEFAULT 0,
  strat_foreign_big_sell TINYINT NOT NULL DEFAULT 0,
  strat_co_sell TINYINT NOT NULL DEFAULT 0,

  -- 停損（signals 產生的參考值；回測仍可用固定 -10%）
  stop_loss_side VARCHAR(8) NULL,
  stop_loss_price DOUBLE NULL,
  stop_loss_pct DOUBLE NULL,

  -- 動能教學：進出場（v1）
  primary_strategy VARCHAR(64) NULL,
  entry_long TINYINT NOT NULL DEFAULT 0,
  exit_long TINYINT NOT NULL DEFAULT 0,
  entry_short TINYINT NOT NULL DEFAULT 0,
  exit_short TINYINT NOT NULL DEFAULT 0,

  PRIMARY KEY (stock_id, trading_date),
  INDEX idx_trade_date (trading_date),
  INDEX idx_score (trading_date, score),
  INDEX idx_score_long (trading_date, score_long),
  INDEX idx_score_short (trading_date, score_short),
  INDEX idx_entry_long (trading_date, entry_long, score),
  INDEX idx_entry_short (trading_date, entry_short, score),
  INDEX idx_prob_kpi_5d_2pct_touch (trading_date, prob_kpi_5d_2pct_touch)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

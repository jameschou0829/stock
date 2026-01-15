-- 教學版：多方策略｜量大動能（多方）
-- 需求：
-- - stock_daily 新增 turnover（成交金額）
-- - stock_signals_v2 新增 momentum_score / turnover / yesterday_turnover

-- ============================================================
-- 1) stock_daily：成交金額（TWD）
-- ============================================================
ALTER TABLE stock_daily
  ADD COLUMN turnover BIGINT NULL COMMENT '成交金額(TWD)' AFTER volume;

-- 建議：若 stock_daily 尚未有 PRIMARY KEY(stock_id, trading_date)，請先補（此專案一般已存在）
-- ALTER TABLE stock_daily ADD PRIMARY KEY (stock_id, trading_date);

-- ============================================================
-- 2) stock_signals_v2：教學版動能欄位
-- ============================================================
ALTER TABLE stock_signals_v2
  ADD COLUMN momentum_score DOUBLE NULL COMMENT '教學版 momentum_score（約 0~20；門檻 7.5）',
  ADD COLUMN turnover BIGINT NULL COMMENT '當日成交金額(TWD，debug用)',
  ADD COLUMN yesterday_turnover BIGINT NULL COMMENT '前一交易日成交金額(TWD，策略判斷用)';

-- ============================================================
-- 3) 索引建議（可選，視查詢型態再加）
-- ============================================================
-- 若未來有「按交易日篩出量大動能上榜名單」的查詢，建議加：
-- CREATE INDEX idx_signals_date_strat_vm ON stock_signals_v2 (trading_date, strat_volume_momentum, score);
--
-- 若未來有「按 momentum_score / yesterday_turnover debug」的查詢，建議加：
-- CREATE INDEX idx_signals_date_mom_turn ON stock_signals_v2 (trading_date, momentum_score, yesterday_turnover);


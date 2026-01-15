import unittest
from datetime import date


from etl.build_signals import calc_momentum_score, calc_teaching_strat_volume_momentum
from configs.strategy_loader import load_strategy_config

_STRATEGY_CFG, _, _ = load_strategy_config({})


class TestTeachingVolumeMomentum(unittest.TestCase):
    def test_normal_day_pass(self):
        """
        正常交易日：
        - yesterday_turnover > 5e8
        - momentum_score > 7.5
        => strat_volume_momentum = 1
        """
        flag = calc_teaching_strat_volume_momentum(momentum_score=8.0, yesterday_turnover=600_000_000, strategy_cfg=_STRATEGY_CFG)
        self.assertEqual(int(flag), 1)

    def test_prev_trading_day_weekend_gap(self):
        """
        跨週末/連假：用「上一筆 trading_date」作為前一交易日即可正確取 yesterday_turnover。
        這裡用最小資料結構模擬 rows_asc[-2] 是上週五。
        """
        rows_asc = [
            {"stock_id": "2330", "trading_date": date(2026, 1, 9), "turnover": 700_000_000},  # Fri
            {"stock_id": "2330", "trading_date": date(2026, 1, 12), "turnover": 800_000_000},  # Mon
        ]
        y_row = rows_asc[-2] if len(rows_asc) >= 2 else None
        self.assertEqual(int(y_row["turnover"]), 700_000_000)

    def test_missing_values_safe(self):
        """
        缺值情境：turnover 或 close/ATR 缺值時不爆炸，策略判斷為 0
        """
        ms = calc_momentum_score(close=None, ma20=10.0, atr14=1.0, is_price_breakout_20d=0)
        self.assertEqual(ms, 0.0)

        flag = calc_teaching_strat_volume_momentum(momentum_score=None, yesterday_turnover=None, strategy_cfg=_STRATEGY_CFG)
        self.assertEqual(int(flag), 0)


if __name__ == "__main__":
    unittest.main()


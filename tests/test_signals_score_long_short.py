import unittest

from etl.build_signals import calc_strategies_and_score
from configs.strategy_loader import load_strategy_config

_STRATEGY_CFG, _, _ = load_strategy_config({})


def _base_sig():
    return {
        "close": 100.0,
        "volume": 1000,
        "turnover": 1,
        "yesterday_turnover": 600_000_000,
        "momentum_score": 8.0,
        "ma20": 95.0,
        "ma60": 90.0,
        "atr14": 2.0,
        "above_ma20": 1,
        "above_ma60": 1,
        "is_price_breakout_20d": 1,
        "is_price_breakdown_20d": 0,
        "is_volume_breakout_20d": 1,
        "dist_40d_high_pct": 0.0,
        "is_near_40d_high": 1,
        "is_price_breakout_400d": 0,
        "is_volume_breakout_10d": 0,
        "foreign_net": 0,
        "trust_net": 0,
        "dealer_net": 0,
        "foreign_net_3d": 0,
        "trust_net_3d": 10,
        "trust_net_5d": 0,
        "trust_buy_streak": 0,
        "is_foreign_first_buy": 1,
        "is_trust_first_buy": 1,
        "is_co_buy": 1,
        "is_foreign_buy_3d": 0,
        "is_trust_buy_3d": 1,
        "is_trust_buy_5d": 0,
        "is_foreign_first_sell": 0,
        "is_trust_first_sell": 0,
        "is_foreign_sell_3d": 0,
        "is_trust_sell_5d": 0,
        "is_co_sell": 0,
        "margin_balance": 0,
        "margin_balance_chg_3d": 0,
        "is_margin_risk": 0,
        "branch_top5_buy_ratio": 0.0,
        "branch_top5_net_ratio": 0.0,
        "is_branch_concentration": 0,
    }


class TestSignalsScoreLongShort(unittest.TestCase):
    def test_score_long_and_entry_long(self):
        sig = _base_sig()
        out = calc_strategies_and_score("bull", sig, _STRATEGY_CFG)
        self.assertGreater(int(out.get("score_long", 0)), 0)
        self.assertEqual(int(out.get("entry_long", 0)), 1)

    def test_score_short_not_negative_mixed(self):
        sig = _base_sig()
        # 做成空方條件強：跌破 + 土洋同賣
        sig["close"] = 90.0
        sig["ma20"] = 100.0
        sig["ma60"] = 105.0
        sig["above_ma20"] = 0
        sig["above_ma60"] = 0
        sig["is_price_breakdown_20d"] = 1
        sig["is_co_sell"] = 1
        sig["is_foreign_first_sell"] = 1
        sig["is_trust_first_sell"] = 1
        out = calc_strategies_and_score("bear", sig, _STRATEGY_CFG)
        self.assertGreaterEqual(int(out.get("score_short", 0)), 0)
        self.assertEqual(int(out.get("entry_short", 0)), 1)

    def test_market_regime_side_weight_does_not_weaken_short_in_bear(self):
        sig = _base_sig()
        sig["close"] = 90.0
        sig["ma20"] = 100.0
        sig["ma60"] = 105.0
        sig["above_ma20"] = 0
        sig["above_ma60"] = 0
        sig["is_price_breakdown_20d"] = 1
        sig["is_co_sell"] = 1
        sig["is_foreign_first_sell"] = 1
        sig["is_trust_first_sell"] = 1
        out_bear = calc_strategies_and_score("bear", dict(sig), _STRATEGY_CFG)
        out_bull = calc_strategies_and_score("bull", dict(sig), _STRATEGY_CFG)
        self.assertGreaterEqual(int(out_bear.get("score_short", 0)), int(out_bull.get("score_short", 0)))


if __name__ == "__main__":
    unittest.main()


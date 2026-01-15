import unittest


from etl.build_signals import calc_strategies_and_score
from configs.strategy_loader import load_strategy_config

_STRATEGY_CFG, _, _ = load_strategy_config({})


def _base_sig():
    # calc_strategies_and_score 需要的最低欄位集合
    return {
        "close": 100.0,
        "ma20": 100.0,
        "ma60": 100.0,
        "atr14": 1.0,
        "above_ma20": 0,
        "above_ma60": 0,
        "is_price_breakout_20d": 0,
        "is_price_breakdown_20d": 0,
        "is_volume_breakout_20d": 0,
        "is_price_breakout_400d": 0,
        "is_volume_breakout_10d": 0,
        "dist_40d_high_pct": None,
        "is_near_40d_high": 0,
        "turnover": 1,
        "yesterday_turnover": 0,
        "momentum_score": 0.0,
        "foreign_net": 0,
        "trust_net": 0,
        "dealer_net": 0,
        "foreign_net_3d": 0,
        "trust_net_3d": 0,
        "trust_net_5d": 0,
        "trust_buy_streak": 0,
        "is_foreign_first_buy": 0,
        "is_trust_first_buy": 0,
        "is_co_buy": 0,
        "is_foreign_buy_3d": 0,
        "is_trust_buy_3d": 0,
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


class TestStrategy3_4_5Long(unittest.TestCase):
    def test_strategy3_trust_momentum_buy(self):
        # 動能>7.5 + 昨日成交金額>5e8 + 投信連買>=2
        sig = _base_sig()
        sig["momentum_score"] = 8.0
        sig["yesterday_turnover"] = 600_000_000
        sig["trust_buy_streak"] = 2
        out = calc_strategies_and_score("range", sig, _STRATEGY_CFG)
        self.assertEqual(int(out.get("strat_trust_momentum_buy", 0)), 1)

    def test_strategy4_foreign_first_buy(self):
        sig = _base_sig()
        sig["momentum_score"] = 9.0
        sig["yesterday_turnover"] = 900_000_000
        sig["is_foreign_first_buy"] = 1
        out = calc_strategies_and_score("range", sig, _STRATEGY_CFG)
        self.assertEqual(int(out.get("strat_foreign_big_buy", 0)), 1)

    def test_strategy5_co_buy_threshold(self):
        sig = _base_sig()
        sig["momentum_score"] = 6.1
        sig["yesterday_turnover"] = 600_000_000
        sig["is_co_buy"] = 1
        sig["foreign_net"] = 60_000_000
        sig["trust_net"] = 1
        out = calc_strategies_and_score("range", sig, _STRATEGY_CFG)
        self.assertEqual(int(out.get("strat_co_buy", 0)), 1)


if __name__ == "__main__":
    unittest.main()


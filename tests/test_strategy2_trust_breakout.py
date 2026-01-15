import unittest
from datetime import date, timedelta


from etl.build_signals import calc_kbar_signals, calc_strategies_and_score
from configs.strategy_loader import load_strategy_config

_STRATEGY_CFG, _, _ = load_strategy_config({})


def _base_sig():
    # calc_strategies_and_score 會讀不少欄位；這裡給足預設，避免 KeyError
    return {
        "close": 100.0,
        "volume": 1000,
        "turnover": 1,
        "yesterday_turnover": 0,
        "momentum_score": 0.0,
        "ma20": 100.0,
        "ma60": 100.0,
        "atr14": 1.0,
        "above_ma20": 0,
        "above_ma60": 0,
        "is_price_breakout_20d": 0,
        "is_price_breakdown_20d": 0,
        "is_volume_breakout_20d": 0,
        "dist_40d_high_pct": None,
        "is_near_40d_high": 0,
        "foreign_net": 0,
        "trust_net": 0,
        "dealer_net": 0,
        "foreign_net_3d": 0,
        "trust_net_3d": 0,
        "trust_net_5d": 0,
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


class TestStrategy2TrustBreakout(unittest.TestCase):
    def test_kbar_dist_40d_high_near(self):
        """
        40日最高價乖離介於(含)2%~-2%：is_near_40d_high=1
        """
        start = date(2026, 1, 1)
        daily_desc = []
        # 造 45 天資料（new->old）
        for i in range(45):
            d = start + timedelta(days=i)
            high = 100.0  # 40日最高固定 100
            low = 90.0
            close = 99.0  # 距離最高價 -1%
            daily_desc.insert(
                0,
                {"trading_date": d, "open": close, "high": high, "low": low, "close": close, "volume": 1000},
            )

        sig = calc_kbar_signals(daily_desc)
        self.assertEqual(int(sig.get("is_near_40d_high", 0)), 1)
        self.assertTrue(sig.get("dist_40d_high_pct") is not None)
        self.assertAlmostEqual(float(sig["dist_40d_high_pct"]), -1.0, places=6)

    def test_strategy2_flag(self):
        """
        策略2（投信剛買準突破）：
        - is_near_40d_high=1
        - yesterday_turnover>5e8
        - trust_net_3d>0
        """
        sig = _base_sig()
        sig["is_near_40d_high"] = 1
        sig["yesterday_turnover"] = 600_000_000
        sig["trust_net_3d"] = 10

        out = calc_strategies_and_score("range", sig, _STRATEGY_CFG)
        self.assertEqual(int(out.get("strat_trust_breakout", 0)), 1)

    def test_strategy2_fail_when_turnover_low(self):
        sig = _base_sig()
        sig["is_near_40d_high"] = 1
        sig["yesterday_turnover"] = 100_000_000
        sig["trust_net_3d"] = 10
        out = calc_strategies_and_score("range", sig, _STRATEGY_CFG)
        self.assertEqual(int(out.get("strat_trust_breakout", 0)), 0)


if __name__ == "__main__":
    unittest.main()


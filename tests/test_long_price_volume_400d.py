import unittest
from datetime import date, timedelta


from etl.build_signals import calc_kbar_signals, calc_strategies_and_score


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
        "is_price_breakout_400d": 0,
        "is_volume_breakout_10d": 0,
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


class TestLongPriceVolumeNewHigh400d(unittest.TestCase):
    def test_kbar_breakout_400d_and_vol_10d(self):
        """
        多方策略｜價量創新高（PDF OCR）：
        - 價創400日新高
        - 成交量創10日新高
        """
        start = date(2024, 1, 1)
        daily_desc = []

        # 先造 410 天資料（new->old），確保有 401+ 的 close 可算 400d breakout
        # - 前 409 天 close 都是 100
        # - 最後一天（today）close=101 -> 突破
        # - 前 10 日 volume 最大 1000；today volume=1200 -> 10d 量突破
        total_days = 410
        for i in range(total_days):
            d = start + timedelta(days=i)
            if i == total_days - 1:
                close = 101.0
                vol = 1200
            else:
                close = 100.0
                vol = 1000
            daily_desc.insert(
                0,
                {
                    "trading_date": d,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": vol,
                },
            )

        ks = calc_kbar_signals(daily_desc)
        self.assertEqual(int(ks.get("is_price_breakout_400d", 0)), 1)
        self.assertEqual(int(ks.get("is_volume_breakout_10d", 0)), 1)

    def test_strategy_flag(self):
        """
        strat_price_volume_new_high = 1 當：
        - is_price_breakout_400d=1
        - is_volume_breakout_10d=1
        - yesterday_turnover>5e8
        """
        sig = _base_sig()
        sig["is_price_breakout_400d"] = 1
        sig["is_volume_breakout_10d"] = 1
        sig["yesterday_turnover"] = 600_000_000
        out = calc_strategies_and_score("range", sig)
        self.assertEqual(int(out.get("strat_price_volume_new_high", 0)), 1)

    def test_strategy_fail_when_turnover_low(self):
        sig = _base_sig()
        sig["is_price_breakout_400d"] = 1
        sig["is_volume_breakout_10d"] = 1
        sig["yesterday_turnover"] = 100_000_000
        out = calc_strategies_and_score("range", sig)
        self.assertEqual(int(out.get("strat_price_volume_new_high", 0)), 0)


if __name__ == "__main__":
    unittest.main()


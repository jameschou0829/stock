import unittest


from etl.build_signals import calc_kbar_signals


def _make_daily_desc_from_asc(rows_asc):
    """
    build_signals.py 的 calc_kbar_signals() 期待 daily_desc 是「新→舊」
    """
    return list(reversed(rows_asc))


class TestKbarRobustness(unittest.TestCase):
    def test_ma_window_contains_none(self):
        # 準備 25 天資料（舊→新），其中 MA20 視窗內故意塞一個 None
        rows_asc = []
        for i in range(25):
            rows_asc.append(
                {
                    "close": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "volume": 1000 + i,
                }
            )
        # 讓最近 20 天視窗中含 None（例如倒數第 10 天 close 缺值）
        rows_asc[-10]["close"] = None

        sig = calc_kbar_signals(_make_daily_desc_from_asc(rows_asc))
        self.assertIsNone(sig.get("ma20"))
        self.assertEqual(int(sig.get("above_ma20", 0)), 0)
        self.assertEqual(int(sig.get("is_price_breakout_20d", 0)), 0)

    def test_close_zero_safe(self):
        # close=0（異常值）：當日指標不可計算，且不得爆炸
        rows_asc = []
        for i in range(30):
            rows_asc.append(
                {
                    "close": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "volume": 1000 + i,
                }
            )
        rows_asc[-1]["close"] = 0.0
        sig = calc_kbar_signals(_make_daily_desc_from_asc(rows_asc))
        self.assertEqual(sig.get("close"), 0.0)
        self.assertIsNone(sig.get("ma20"))
        self.assertIsNone(sig.get("atr14"))
        self.assertEqual(int(sig.get("is_volume_breakout_20d", 0)), 0)

    def test_insufficient_length(self):
        # 長度不足 <20/<14：MA/ATR/突破都不可計算，不得爆炸
        rows_asc = []
        for i in range(10):
            rows_asc.append(
                {
                    "close": 100.0 + i,
                    "high": 101.0 + i,
                    "low": 99.0 + i,
                    "volume": 1000 + i,
                }
            )
        sig = calc_kbar_signals(_make_daily_desc_from_asc(rows_asc))
        self.assertIsNone(sig.get("ma20"))
        self.assertIsNone(sig.get("atr14"))
        self.assertEqual(int(sig.get("is_price_breakout_20d", 0)), 0)


if __name__ == "__main__":
    unittest.main()


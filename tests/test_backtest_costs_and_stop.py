import unittest
from datetime import date

from backtest.engine import _trade_costs, _stop_price, _hit_stop


class TestBacktestCostsAndStop(unittest.TestCase):
    def test_costs_long_entry_exit(self):
        notional = 100000.0
        c_in = _trade_costs(notional, side="long", action="entry", commission_bps=10, tax_bps=30, slippage_bps=5)
        c_out = _trade_costs(notional, side="long", action="exit", commission_bps=10, tax_bps=30, slippage_bps=5)
        self.assertGreater(c_in, 0.0)
        self.assertGreater(c_out, c_in)  # exit 多了稅

    def test_costs_short_entry_has_tax_exit_no_tax(self):
        notional = 100000.0
        c_in = _trade_costs(notional, side="short", action="entry", commission_bps=10, tax_bps=30, slippage_bps=5)
        c_out = _trade_costs(notional, side="short", action="exit", commission_bps=10, tax_bps=30, slippage_bps=5)
        self.assertGreater(c_in, c_out)  # entry 有稅

    def test_stop_price_long_short(self):
        self.assertAlmostEqual(_stop_price(100.0, side="long", stop_loss_pct=0.1), 90.0)
        self.assertAlmostEqual(_stop_price(100.0, side="short", stop_loss_pct=0.1), 110.0)

    def test_hit_stop_gap(self):
        # long：開盤直接跌破 -> 用 open 成交
        bar = {"open": 89.0, "high": 92.0, "low": 88.0, "close": 90.0}
        self.assertAlmostEqual(_hit_stop(bar, side="long", stop_px=90.0), 89.0)
        # short：開盤直接站上停損 -> 用 open 成交
        bar2 = {"open": 111.0, "high": 112.0, "low": 108.0, "close": 110.0}
        self.assertAlmostEqual(_hit_stop(bar2, side="short", stop_px=110.0), 111.0)


if __name__ == "__main__":
    unittest.main()


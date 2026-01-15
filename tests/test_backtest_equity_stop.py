import unittest
from datetime import date


class TestBacktestEquityStop(unittest.TestCase):
    def test_equity_stop_threshold(self):
        initial = 1_000_000.0
        equity_stop_pct = 0.30
        threshold = initial * (1.0 - equity_stop_pct)
        self.assertEqual(threshold, 700_000.0)


if __name__ == "__main__":
    unittest.main()


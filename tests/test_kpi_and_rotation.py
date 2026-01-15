import unittest

from backtest.kpi import KpiConfig, init_kpi_state, update_kpi_state
from backtest.engine import evaluate_rotation


class TestKpiAndRotation(unittest.TestCase):
    def test_kpi_touch_pass(self):
        cfg = KpiConfig(
            horizon_days=5,
            target_return=0.02,
            price_source="high",
            trigger_mode="touch",
            fail_action="watchlist",
            grace_days=0,
            require_profit_days=0,
        )
        state = init_kpi_state()
        entry_px = 100.0
        highs = [101.0, 101.5, 102.0, 101.0, 101.0]
        closes = [100.5, 101.0, 101.2, 100.8, 100.5]
        for hi, cl in zip(highs, closes):
            state = update_kpi_state(
                state=state,
                cfg=cfg,
                side="long",
                entry_px=entry_px,
                price_high=hi,
                price_close=cl,
            )
        self.assertTrue(state.get("passed"))

    def test_kpi_end_fail(self):
        cfg = KpiConfig(
            horizon_days=5,
            target_return=0.02,
            price_source="close",
            trigger_mode="end",
            fail_action="watchlist",
            grace_days=0,
            require_profit_days=0,
        )
        state = init_kpi_state()
        entry_px = 100.0
        closes = [100.5, 101.0, 101.5, 101.8, 101.9]  # 最終未達 2%
        for cl in closes:
            state = update_kpi_state(
                state=state,
                cfg=cfg,
                side="long",
                entry_px=entry_px,
                price_high=cl,
                price_close=cl,
            )
        self.assertTrue(state.get("failed"))

    def test_rotation_trigger_reason(self):
        should_rotate, reason = evaluate_rotation(
            best_edge=1.0,
            worst_edge=0.9,
            worst_failed=True,
            worst_momentum=False,
            delta_edge=0.03,
            requires_fail=True,
        )
        self.assertTrue(should_rotate)
        self.assertEqual(reason, "rotation")


if __name__ == "__main__":
    unittest.main()

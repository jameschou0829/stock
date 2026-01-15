import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs import settings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--side", default="long", choices=["long", "short", "both"])
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--holding-days", type=int, default=5)
    ap.add_argument("--min-abs-score", type=int, default=0)
    ap.add_argument("--use-stop-loss", action="store_true")
    ap.add_argument("--use-entry-exit-signals", action="store_true", help="用 signals 的 entry/exit 規則（可調 entry_min_score）")
    ap.add_argument("--entry-min-score", type=int, default=getattr(settings, "SIGNALS_ENTRY_MIN_SCORE", 25))
    ap.add_argument("--exit-no-momentum-days", type=int, default=2, help="沒量沒動能連續 N 天則出場（holding_days=0 時特別有用）")
    ap.add_argument("--calendar-stock-id", default=getattr(settings, "MARKET_PROXY_STOCK_ID", "0050"))
    args = ap.parse_args()

    from backtest.engine import BacktestConfig, run_backtest, parse_date

    cfg = BacktestConfig(
        start=parse_date(args.start),
        end=parse_date(args.end),
        side=args.side,
        top_n=int(args.top_n),
        holding_days=int(args.holding_days),
        min_abs_score=int(args.min_abs_score),
        entry_min_score=int(args.entry_min_score),
        use_stop_loss=bool(args.use_stop_loss),
        use_entry_exit_signals=bool(args.use_entry_exit_signals),
        exit_no_momentum_days=int(args.exit_no_momentum_days),
        calendar_stock_id=str(args.calendar_stock_id),
    )
    out = run_backtest(cfg)
    s = out["summary"]

    print("=== Backtest Summary ===")
    print(f"range: {args.start} -> {args.end}")
    print(f"side: {cfg.side} | top_n={cfg.top_n} | holding_days={cfg.holding_days}")
    print(f"use_stop_loss={cfg.use_stop_loss} | min_abs_score={cfg.min_abs_score} | entry_min_score={cfg.entry_min_score}")
    print(f"trades: {s['trades']} | win_rate={s['win_rate']:.2%} | avg_trade_ret={s['avg_trade_ret']:.2%}")
    print(f"equity_end: {s['equity_end']:.4f} | total_ret={s['total_ret']:.2%} | max_drawdown={s['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()


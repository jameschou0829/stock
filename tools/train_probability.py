import argparse
from datetime import datetime

from ml.probability import train_and_update_probabilities


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--train-end", required=False, help="YYYY-MM-DD (train until this date, predict (train_end, end])")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    train_end = datetime.strptime(args.train_end, "%Y-%m-%d").date() if args.train_end else None
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    n = train_and_update_probabilities(start=start, end=end, train_end=train_end)
    print(f"[OK] updated probabilities: {n} rows (pred range: ({train_end or end}, {end}])")


if __name__ == "__main__":
    main()

from __future__ import annotations

import subprocess
import sys

STAGES = [
    "cfb_quant.live.market_join_v2",
    "cfb_quant.live.market_probability_v2",
    "cfb_quant.live.market_probability_calibrated",
    "cfb_quant.live.market_uncertainty",
    "cfb_quant.live.market_role_audit",
    "cfb_quant.live.market_shortlist",
    "cfb_quant.live.market_shortlist_audit",
    "cfb_quant.live.final_review",
    "cfb_quant.live.final_bet_report",
    "cfb_quant.live.final_portfolio",
]

def main():
    print("=" * 90)
    print("CFB LIVE PLAYER PROP MARKET PIPELINE")
    print("=" * 90)

    for number, module in enumerate(STAGES, start=1):
        print()
        print(f"[{number}/{len(STAGES)}] Running {module}")
        print("-" * 90)

        result = subprocess.run(
            [sys.executable, "-m", module],
            check=False,
        )

        if result.returncode != 0:
            print()
            print(f"PIPELINE FAILED AT: {module}")
            sys.exit(result.returncode)

    print()
    print("=" * 90)
    print("PIPELINE COMPLETE")
    print("=" * 90)
    print("Final Top-5: reports/final_portfolio_2026_week_4_v0_1.csv")
    print("Final Top-10: reports/final_bet_report_2026_week_4_v0_1.csv")

if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL = Path("reports/player_prop_holdout_predictions_2025_v0_1.csv")
MARKET = Path("reports/market_join_2026_week_4_v0_2.csv")
OUTPUT = Path("reports/market_probability_2026_week_4_v0_1.csv")
SUMMARY = Path("reports/market_probability_summary_2026_week_4_v0_1.csv")

def american_implied_probability(odds):
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def decimal_odds(odds):
    odds = float(odds)
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    return 1.0 + odds / 100.0

def probability_to_american(p):
    p = float(np.clip(p, 0.0001, 0.9999))
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p

def empirical_over_probability(residuals, threshold):
    residuals = np.sort(np.asarray(residuals, dtype=float))
    n = len(residuals)

    if n == 0:
        return np.nan

    above = n - np.searchsorted(residuals, threshold, side="right")

    # Half-count smoothing prevents exact 0% / 100% probabilities.
    return (above + 0.5) / (n + 1.0)

def expected_value(probability, odds):
    dec = decimal_odds(odds)
    return probability * (dec - 1.0) - (1.0 - probability)

def main():
    historical = pd.read_csv(HISTORICAL)
    market = pd.read_csv(MARKET)

    historical["residual"] = pd.to_numeric(historical["residual"], errors="coerce")
    historical = historical.dropna(subset=["target", "residual"])

    residual_map = {
        target: group["residual"].to_numpy(dtype=float)
        for target, group in historical.groupby("target")
    }

    rows = []

    for _, row in market.iterrows():
        target = row["target"]
        projection = float(row["projection_blended"])
        line = float(row["line"])
        over_odds = float(row["over_odds"])
        under_odds = float(row["under_odds"])

        residuals = residual_map.get(target, np.array([], dtype=float))

        # actual = projection + residual
        # OVER occurs when residual > line - projection
        residual_threshold = line - projection

        p_over = empirical_over_probability(residuals, residual_threshold)
        p_under = 1.0 - p_over

        implied_over = american_implied_probability(over_odds)
        implied_under = american_implied_probability(under_odds)
        hold = implied_over + implied_under

        no_vig_over = implied_over / hold
        no_vig_under = implied_under / hold

        edge_over = p_over - no_vig_over
        edge_under = p_under - no_vig_under

        ev_over = expected_value(p_over, over_odds)
        ev_under = expected_value(p_under, under_odds)

        if ev_over >= ev_under:
            raw_best_side = "OVER"
            raw_best_probability = p_over
            raw_best_no_vig_probability = no_vig_over
            raw_best_edge = edge_over
            raw_best_ev = ev_over
            raw_best_odds = over_odds
        else:
            raw_best_side = "UNDER"
            raw_best_probability = p_under
            raw_best_no_vig_probability = no_vig_under
            raw_best_edge = edge_under
            raw_best_ev = ev_under
            raw_best_odds = under_odds

        record = row.to_dict()
        record.update({
            "historical_residual_rows": len(residuals),
            "residual_threshold": residual_threshold,
            "historical_residual_mean": float(np.mean(residuals)) if len(residuals) else np.nan,
            "historical_residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else np.nan,
            "model_probability_over": p_over,
            "model_probability_under": p_under,
            "market_implied_over": implied_over,
            "market_implied_under": implied_under,
            "market_hold": hold - 1.0,
            "no_vig_probability_over": no_vig_over,
            "no_vig_probability_under": no_vig_under,
            "probability_edge_over": edge_over,
            "probability_edge_under": edge_under,
            "ev_over": ev_over,
            "ev_under": ev_under,
            "fair_odds_over": probability_to_american(p_over),
            "fair_odds_under": probability_to_american(p_under),
            "raw_best_side": raw_best_side,
            "raw_best_probability": raw_best_probability,
            "raw_best_no_vig_probability": raw_best_no_vig_probability,
            "raw_best_probability_edge": raw_best_edge,
            "raw_best_ev": raw_best_ev,
            "raw_best_odds": raw_best_odds,
        })

        rows.append(record)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["raw_best_ev", "raw_best_probability_edge"],
        ascending=[False, False],
    ).reset_index(drop=True)

    out.to_csv(OUTPUT, index=False)

    summary = (
        out.groupby("target")
        .agg(
            market_rows=("target", "size"),
            historical_rows=("historical_residual_rows", "first"),
            mean_abs_projection_edge=("projection_edge", lambda s: float(np.mean(np.abs(s)))),
            mean_raw_best_ev=("raw_best_ev", "mean"),
            max_raw_best_ev=("raw_best_ev", "max"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY, index=False)

    print("=" * 120)
    print("CFB PLAYER PROP RAW PROBABILITY + EV")
    print("=" * 120)
    print(f"Market rows: {len(out):,}")
    print(f"Targets: {out['target'].nunique()}")
    print(f"Missing probabilities: {out['raw_best_probability'].isna().sum()}")
    print()

    display_cols = [
        "away_team",
        "home_team",
        "player_name_model",
        "target",
        "line",
        "projection_blended",
        "raw_best_side",
        "raw_best_odds",
        "raw_best_probability",
        "raw_best_no_vig_probability",
        "raw_best_probability_edge",
        "raw_best_ev",
        "model_disagreement",
    ]

    print("TOP RAW EV RESULTS")
    print("-" * 120)
    print(
        out[display_cols]
        .head(40)
        .round(4)
        .to_string(index=False)
    )

    print()
    print("TARGET SUMMARY")
    print("-" * 120)
    print(summary.round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)
    print("Saved:", SUMMARY)
    print()
    print("IMPORTANT: raw EV only - no uncertainty, role, sample-size, or stability penalty applied yet.")

if __name__ == "__main__":
    main()

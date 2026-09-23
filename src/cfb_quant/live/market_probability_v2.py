from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HISTORICAL = Path("reports/player_prop_holdout_predictions_2025_v0_1.csv")
MARKET = Path("reports/market_join_2026_week_4_v0_2.csv")
OUTPUT = Path("reports/market_probability_2026_week_4_v0_2.csv")
SUMMARY = Path("reports/market_probability_summary_2026_week_4_v0_2.csv")

NEIGHBORS = {
    "pass_attempts": 600,
    "completions": 600,
    "passing_yards": 600,
    "rush_attempts": 1000,
    "rushing_yards": 1000,
    "receptions": 1400,
    "receiving_yards": 1400,
}

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

def expected_value(probability, odds):
    dec = decimal_odds(odds)
    return probability * (dec - 1.0) - (1.0 - probability)

def local_residuals(group, projection, k):
    g = group.copy()
    g["projection_distance"] = np.abs(g["prediction_blended"] - projection)
    g = g.nsmallest(min(k, len(g)), "projection_distance")
    return g

def empirical_over_probability(residuals, threshold):
    residuals = np.sort(np.asarray(residuals, dtype=float))
    n = len(residuals)
    if n == 0:
        return np.nan
    above = n - np.searchsorted(residuals, threshold, side="right")
    return (above + 0.5) / (n + 1.0)

def main():
    historical = pd.read_csv(HISTORICAL)
    market = pd.read_csv(MARKET)

    historical["prediction_blended"] = pd.to_numeric(historical["prediction_blended"], errors="coerce")
    historical["residual"] = pd.to_numeric(historical["residual"], errors="coerce")
    historical = historical.dropna(subset=["target", "prediction_blended", "residual"])

    historical_by_target = {
        target: group.copy()
        for target, group in historical.groupby("target")
    }

    rows = []

    for _, row in market.iterrows():
        target = row["target"]
        projection = float(row["projection_blended"])
        line = float(row["line"])
        over_odds = float(row["over_odds"])
        under_odds = float(row["under_odds"])

        group = historical_by_target[target]
        k = NEIGHBORS[target]
        local = local_residuals(group, projection, k)
        residuals = local["residual"].to_numpy(dtype=float)

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
            best_side = "OVER"
            best_probability = p_over
            best_no_vig = no_vig_over
            best_edge = edge_over
            best_ev = ev_over
            best_odds = over_odds
        else:
            best_side = "UNDER"
            best_probability = p_under
            best_no_vig = no_vig_under
            best_edge = edge_under
            best_ev = ev_under
            best_odds = under_odds

        record = row.to_dict()
        record.update({
            "local_residual_rows": len(local),
            "local_projection_min": float(local["prediction_blended"].min()),
            "local_projection_max": float(local["prediction_blended"].max()),
            "local_projection_mean": float(local["prediction_blended"].mean()),
            "local_projection_mean_abs_distance": float(local["projection_distance"].mean()),
            "local_residual_mean": float(local["residual"].mean()),
            "local_residual_std": float(local["residual"].std(ddof=1)),
            "local_residual_mae": float(local["residual"].abs().mean()),
            "residual_threshold": residual_threshold,
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
            "raw_best_side": best_side,
            "raw_best_probability": best_probability,
            "raw_best_no_vig_probability": best_no_vig,
            "raw_best_probability_edge": best_edge,
            "raw_best_ev": best_ev,
            "raw_best_odds": best_odds,
        })

        rows.append(record)

    out = pd.DataFrame(rows)
    out = out.sort_values(["raw_best_ev","raw_best_probability_edge"], ascending=[False,False]).reset_index(drop=True)
    out.to_csv(OUTPUT, index=False)

    summary = out.groupby("target").agg(
        market_rows=("target","size"),
        local_rows=("local_residual_rows","first"),
        mean_local_residual_std=("local_residual_std","mean"),
        mean_raw_best_ev=("raw_best_ev","mean"),
        max_raw_best_ev=("raw_best_ev","max"),
    ).reset_index()
    summary.to_csv(SUMMARY, index=False)

    print("=" * 120)
    print("CFB PLAYER PROP LOCAL RESIDUAL PROBABILITY + EV")
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
        "local_residual_std",
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
    print(out[display_cols].head(40).round(4).to_string(index=False))
    print()
    print("TARGET SUMMARY")
    print("-" * 120)
    print(summary.round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)
    print("Saved:", SUMMARY)
    print()
    print("IMPORTANT: local residual calibration only - no uncertainty or role-stability penalty yet.")

if __name__ == "__main__":
    main()

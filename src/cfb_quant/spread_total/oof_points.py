"""Only generate chronological RIDGE team-points OOF forecasts.

Uses existing feature manifest and the same ridge factory as CFB-Quant.
Does not read betting odds, and never fits on a prediction season's outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cfb_quant.models.matchup_tournament import _make_ridge, _target_mask, _to_numeric_frame


def build_points_oof(data: pd.DataFrame, features: list[str], *,
                     first_year: int = 2015, last_year: int = 2025,
                     min_train_year: int = 2014) -> pd.DataFrame:
    if any(c not in data.columns for c in features):
        raise ValueError("Missing manifest features")
    if "points" in features or "opponent_points" in features:
        raise ValueError("Current-game scoring target leaked into input features")
    eligible = _target_mask(data, "points")
    all_out = []
    for year in range(first_year, last_year + 1):
        train = data.loc[eligible & data.season.ge(min_train_year) & data.season.lt(year)].copy()
        test = data.loc[eligible & data.season.eq(year)].copy()
        if train.empty or test.empty:
            print(f"SKIP year={year}: train={len(train)} test={len(test)}")
            continue
        if int(train.season.max()) >= year:
            raise ValueError("Chronology violation")
        x_train = _to_numeric_frame(train, features)
        x_test = _to_numeric_frame(test, features)
        y_train = pd.to_numeric(train.points, errors="coerce")
        mdl = _make_ridge("points")
        mdl.fit(x_train, y_train)
        pred = np.clip(np.asarray(mdl.predict(x_test), dtype=float), 0.0, 100.0)
        out = test[["game_id", "season", "week", "team", "home_away"]].copy()
        out["target"] = "points"
        out["model"] = "ridge"
        out["prediction"] = pred
        out["actual"] = pd.to_numeric(test.points, errors="coerce").to_numpy()
        out["train_end_season"] = int(train.season.max())
        all_out.append(out)
        print(f"{year}: predictions={len(out)}, training through {int(train.season.max())}")
    if not all_out:
        raise RuntimeError("No chronological point forecasts generated")
    oof = pd.concat(all_out, ignore_index=True)
    if oof.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate team-game in OOF results")
    return oof


def main() -> None:
    ap = argparse.ArgumentParser(description="Build season-blocked OOF point forecasts using existing CFB-Quant ridge model")
    ap.add_argument("--features", default="data/processed/cfb_matchup_features_v0_1.parquet")
    ap.add_argument("--manifest", default="reports/cfb_matchup_feature_manifest_v0_1.json")
    ap.add_argument("--out", default="data/processed/cfb_spread_total_points_oof_v1.parquet")
    ap.add_argument("--first-year", type=int, default=2015)
    ap.add_argument("--last-year", type=int, default=2025)
    args = ap.parse_args()
    data = pd.read_parquet(args.features)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    out = build_points_oof(data, list(manifest["feature_columns"]), first_year=args.first_year,
                           last_year=args.last_year)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    print(f"Saved {len(out)} forecasts to {dest}")

if __name__ == "__main__":
    main()

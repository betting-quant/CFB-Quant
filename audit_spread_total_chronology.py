"""Read-only chronology audit for the CFB-Quant matchup feature pipeline.

Does not alter data, request API data, train models, or assert point-in-time validity.
Usage from CFB-Quant root:
    python audit_spread_total_chronology.py
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEASON_TYPE_ORDER = {"regular": 0, "postseason": 1, "spring_regular": 2, "spring_postseason": 3}
KICKOFF_COLUMNS = (
    "start_date", "start_date_utc", "game_start_utc", "scheduled_start_utc",
    "kickoff_utc", "kickoff", "start_time", "start_time_utc", "game_date",
    "date", "startDate", "startTime", "kickoff_time", "start_datetime",
)
SNAPSHOT_COLUMNS = (
    "retrieved_at_utc", "as_of_utc", "available_at_utc", "source_snapshot_utc",
    "ingested_at_utc", "fetched_at", "loaded_at", "created_at",
)


def _norm_type(s: pd.Series) -> pd.Series:
    return s.fillna("unknown").astype(str).str.lower().str.strip()


def detect_date_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None, list[dict]]:
    candidates = []
    parsed_results = []
    for col in KICKOFF_COLUMNS:
        if col not in df.columns:
            continue
        # utc=True compares timezone-aware kickoff timestamps consistently, but
        # timezone-naive sources still require provenance verification.
        parsed = pd.to_datetime(df[col], utc=True, errors="coerce", format="mixed")
        n = int(parsed.notna().sum())
        candidates.append({"column": col, "non_null_parseable": n, "total": len(df)})
        parsed_results.append((col, parsed, n))
    if not parsed_results:
        return None, None, candidates
    best = max(parsed_results, key=lambda t: t[2])
    return (best[0], best[1], candidates) if best[2] else (None, None, candidates)


def chronological_inversions(df: pd.DataFrame) -> pd.DataFrame:
    """Find true kickoff time moving backwards in the existing model's sort order."""
    cols = ["game_id", "season", "week", "season_type", "team", "opponent", "kickoff_utc"]
    if not set(cols).issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(set(cols)-set(df.columns))}")
    out = df[cols].copy()
    out["_season_type_order"] = _norm_type(out["season_type"]).map(SEASON_TYPE_ORDER).fillna(9).astype(int)
    out = out.sort_values(["season", "_season_type_order", "week", "game_id", "team"], kind="mergesort")
    out = out.loc[out.kickoff_utc.notna()].copy()
    out["previous_kickoff_utc"] = out.groupby(["season", "team"], dropna=False)["kickoff_utc"].shift()
    out["previous_game_id"] = out.groupby(["season", "team"], dropna=False)["game_id"].shift()
    out["previous_opponent"] = out.groupby(["season", "team"], dropna=False)["opponent"].shift()
    backward = out["previous_kickoff_utc"].gt(out["kickoff_utc"])
    return out.loc[backward, ["season", "team", "game_id", "opponent", "week", "season_type", "kickoff_utc", "previous_game_id", "previous_opponent", "previous_kickoff_utc"]].reset_index(drop=True)


def classify_missingness(feature_frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    if not isinstance(features, list) or not features:
        raise ValueError("Expected a nonempty manifest feature_columns list")
    if set(features)-set(feature_frame.columns):
        raise ValueError("Missing manifest features")
    out = feature_frame.copy()
    out["season_type_norm"] = _norm_type(out["season_type"])
    w = pd.to_numeric(out["week"], errors="coerce")
    out["phase"] = np.select(
        [out.season_type_norm.eq("regular") & w.le(4),
         out.season_type_norm.eq("regular") & w.gt(4),
         out.season_type_norm.eq("postseason")],
        ["regular_weeks_1_to_4", "regular_week_5_plus", "postseason"],
        default="other_or_unknown",
    )
    missing = out[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).isna()
    rows = []
    for label, idx in out.groupby("phase", dropna=False).groups.items():
        m = missing.loc[idx]
        rows.append({
            "phase": label,
            "team_game_rows": len(m),
            "mean_feature_missing_pct": float(m.to_numpy().mean()*100),
            "median_row_missing_pct": float(m.mean(axis=1).median()*100),
        })
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


def audit(root: Path, outdir: Path, feature_path: str, manifest_path: str, raw_glob: str) -> dict:
    feature_path = root/feature_path
    manifest_path = root/manifest_path
    raw_files = sorted(glob.glob(str(root/raw_glob)))
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = manifest["feature_columns"]
    df = pd.read_parquet(feature_path)
    required = {"game_id", "season", "week", "season_type", "team", "opponent"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature file missing {sorted(required-set(df.columns))}")
    outdir.mkdir(parents=True, exist_ok=True)

    phase_missing = classify_missingness(df, features)
    phase_missing.to_csv(outdir/"missingness_by_real_phase.csv", index=False)
    df = df.copy()
    df["season_type_norm"] = _norm_type(df["season_type"])
    phase_counts = df.groupby(["season_type_norm", "week"], dropna=False).agg(team_game_rows=("game_id", "size"), games=("game_id", "nunique")).reset_index()
    phase_counts.to_csv(outdir/"game_counts_by_season_type_week.csv", index=False)

    repeated = df.loc[df.duplicated(["season", "week", "team"], keep=False)]
    repeated_summary = repeated.groupby(["week", "season_type_norm"]).size().rename("team_game_rows").reset_index()
    repeated_summary.to_csv(outdir/"repeated_week_breakdown.csv", index=False)

    date_source = None
    date_col = None
    date_candidates = {}
    kickoff = None
    if raw_files:
        raw = pd.concat([pd.read_parquet(p) for p in raw_files], ignore_index=True, sort=False)
        if "game_id" not in raw:
            raise ValueError("Raw team-game files do not contain game_id")
        col, dates, candidates = detect_date_column(raw)
        date_candidates["raw_team_games"] = candidates
        if dates is not None:
            raw_dt = pd.DataFrame({"game_id": raw.game_id, "kickoff_utc": dates})
            conflicting = raw_dt.dropna().groupby("game_id")["kickoff_utc"].nunique()
            if conflicting.gt(1).any():
                raise ValueError(f"Different kickoff dates for identical game_id. Sample: {conflicting[conflicting.gt(1)].head(5).index.tolist()}")
            game_dt = (raw_dt.sort_values("kickoff_utc", na_position="last")
                       .drop_duplicates("game_id"))
            merged = df[["game_id", "season", "week", "season_type", "team", "opponent"]].merge(game_dt, on="game_id", how="left", validate="many_to_one")
            kickoff = merged
            date_source, date_col = "raw_team_games", col
    if kickoff is None:
        col, dates, candidates = detect_date_column(df)
        date_candidates["matchup_features"] = candidates
        if dates is not None:
            kickoff = df[["game_id", "season", "week", "season_type", "team", "opponent"]].copy()
            kickoff["kickoff_utc"] = dates
            date_source, date_col = "matchup_features", col

    inversions = pd.DataFrame(columns=["season", "team", "game_id", "opponent", "week", "season_type", "kickoff_utc", "previous_game_id", "previous_opponent", "previous_kickoff_utc"])
    known_game_dates = 0
    unknown_game_dates = int(df.game_id.nunique())
    if kickoff is not None:
        inversions = chronological_inversions(kickoff)
        known_game_dates = int(kickoff.loc[kickoff.kickoff_utc.notna(), "game_id"].nunique())
        unknown_game_dates -= known_game_dates
    inversions.to_csv(outdir/"kickoff_order_inversions.csv", index=False)

    summary = {
        "model_input_game_count": int(df.game_id.nunique()),
        "model_input_team_game_rows": len(df),
        "repeated_team_week_rows": len(repeated),
        "repeated_weeks": sorted(int(v) for v in pd.to_numeric(repeated.week, errors="coerce").dropna().unique()),
        "raw_team_game_files_found": len(raw_files),
        "date_source": date_source,
        "selected_kickoff_column": date_col,
        "kickoff_field_candidates": date_candidates,
        "games_with_parsed_kickoff": known_game_dates,
        "games_without_parsed_kickoff": unknown_game_dates,
        "chronological_inversions_in_model_order": len(inversions),
        "source_snapshot_columns_in_features": sorted(set(SNAPSHOT_COLUMNS) & set(df.columns)),
        "interpretation": (
            "Detected ordering inversions: stop and repair feature chronology, regenerate OOF and revalidate."
            if len(inversions) else
            "Kickoff times unavailable: ordering not verified; schedule timestamps must be added."
            if kickoff is None else
            "No inversions detected among timestamped games; missing dates and source availability still require audit."
        ),
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print("CFB-QUANT CHRONOLOGY AUDIT — READ ONLY")
    print(f"Games: {summary['model_input_game_count']:,} | repeated team/week rows: {len(repeated):,}")
    print("\nMissingness separated by true season type:")
    print(phase_missing.round(2).to_string(index=False))
    print("\nRaw team-game files:", len(raw_files))
    print("Kickoff date column:", f"{date_source}.{date_col}" if date_col else "NONE FOUND")
    print(f"Games with usable kickoff: {known_game_dates:,}/{unknown_game_dates+known_game_dates:,}")
    print("Sort-order inversions detected:", len(inversions))
    print("RESULT:", summary["interpretation"])
    print("Saved reports:", outdir)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".")
    p.add_argument("--outdir", default="reports/spread_total_chronology_audit_v1")
    p.add_argument("--features", default="data/processed/cfb_matchup_features_v0_1.parquet")
    p.add_argument("--manifest", default="reports/cfb_matchup_feature_manifest_v0_1.json")
    p.add_argument("--raw-glob", default="data/processed/team_games_*.parquet")
    a = p.parse_args()
    audit(Path(a.root), Path(a.root)/a.outdir, a.features, a.manifest, a.raw_glob)

if __name__ == "__main__":
    main()

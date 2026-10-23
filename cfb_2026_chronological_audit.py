#!/usr/bin/env python3
"""Read-only, best-effort 2026 chronological audit for archived CFB-Quant forecasts.

Run from repository root with its virtualenv. Historical-model PIT provenance is
NOT certified by this script; archive timestamps and local hashes are inspected.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import re
import sys

import numpy as np
import pandas as pd

SEASON = 2026
ROOT = Path.cwd().resolve()
REPORTS = ROOT / "reports"
SCHEDULE = ROOT / "data/raw/cfbd/games/games_2026.json"

# Three official final scores previously checked in this conversation. These
# can be used if the local raw schedule hasn't been refreshed since Oct 16.
# All other 2026 games require final results in the user's local schedule.
VERIFIED_FALLBACKS = {
    ("middle tennessee", "florida international"): (26, 23),
    ("missouri state", "delaware"): (38, 61),
    ("south alabama", "marshall"): (13, 20),
}
ALIASES = {
    "fiu": "florida international",
    "florida intl": "florida international",
    "florida international": "florida international",
    "middle tennessee state": "middle tennessee",
    "middle tennessee": "middle tennessee",
    "missouri state": "missouri state",
    "delaware": "delaware",
    "south alabama": "south alabama",
    "marshall": "marshall",
}


def norm_team(v: object) -> str:
    s = " ".join(str(v or "").strip().lower().split())
    return ALIASES.get(s, s)


def norm_id(v: object) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return re.sub(r"\.0$", "", str(v).strip())


def utc(v: object):
    try:
        t = pd.to_datetime(v, errors="coerce", utc=True)
        return None if pd.isna(t) else t
    except (TypeError, ValueError, OverflowError):
        return None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def jread(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def list_of_games(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in ("games", "data", "results"):
            if isinstance(obj.get(key), list):
                return obj[key]
    raise ValueError("Could not identify game list in schedule JSON")


def build_results_index():
    by_id, by_teams, stats = {}, {}, Counter()
    if not SCHEDULE.is_file():
        print(f"WARNING: local schedule missing: {SCHEDULE}")
        return by_id, by_teams, stats
    try:
        games = list_of_games(jread(SCHEDULE))
    except Exception as ex:
        print(f"WARNING: schedule unreadable: {ex}")
        return by_id, by_teams, stats
    for g in games:
        if not isinstance(g, dict):
            continue
        stats["schedule_rows"] += 1
        away = norm_team(g.get("awayTeam", g.get("away_team")))
        home = norm_team(g.get("homeTeam", g.get("home_team")))
        gid = norm_id(g.get("id", g.get("game_id")))
        try:
            a = int(g.get("awayPoints", g.get("away_points")))
            h = int(g.get("homePoints", g.get("home_points")))
        except (TypeError, ValueError):
            continue
        complete = g.get("completed", False)
        if isinstance(complete, str):
            complete = complete.strip().lower() == "true"
        if complete is not True:
            continue
        if a < 0 or h < 0 or not away or not home:
            continue
        payload = {"actual_away": a, "actual_home": h,
                   "result_source": "local_cfbd_schedule", "week": g.get("week")}
        if gid:
            by_id[gid] = payload
        by_teams[(away, home)] = payload
        stats["completed_schedule_rows"] += 1
    return by_id, by_teams, stats


def score_for(row: pd.Series, by_id: dict, by_teams: dict):
    key = (norm_team(row["away"]), norm_team(row["home"]))
    by_schedule = by_id.get(norm_id(row["game_id"])) or by_teams.get(key)
    fallback = VERIFIED_FALLBACKS.get(key)
    if by_schedule is not None and fallback is not None:
        if (by_schedule["actual_away"], by_schedule["actual_home"]) != fallback:
            return None, "RESULT_CONFLICT_LOCAL_SCHEDULE_VS_VERIFIED_FALLBACK"
    if by_schedule is not None:
        return by_schedule, None
    if fallback is not None:
        return {"actual_away": fallback[0], "actual_home": fallback[1],
                "result_source": "three_game_verified_fallback", "week": None}, None
    return None, "NO_COMPLETED_SCORE_IN_LOCAL_SCHEDULE"


def is_blank_game_outcome(snapshot_rows: pd.DataFrame) -> bool:
    # Only same-game outcomes: lagged and historical outcome-derived features
    # are expected to contain values.
    for col in ("points", "opponent_points", "actual_margin", "actual_total"):
        if col in snapshot_rows.columns and snapshot_rows[col].notna().any():
            return False
    return True


def scan_archives():
    records, excluded, candidate_paths, archive_report = [], [], [], []
    if not REPORTS.is_dir():
        raise FileNotFoundError(f"Missing reports directory: {REPORTS}")

    # Additional files with unrecognized layouts are inventory only, not scored.
    for candidate in REPORTS.rglob("*.csv"):
        n = candidate.name.lower()
        if n in ("paired_forecasts.csv", "predictions.csv", "live_forecasts.csv"):
            candidate_paths.append(candidate)

    for path in sorted(p for p in candidate_paths if p.name == "paired_forecasts.csv"):
        d = path.parent
        audit_file = d / "audit.json"
        snapshot_path = d / "pregame_features.parquet"
        label = str(path.relative_to(ROOT))
        if not audit_file.is_file() or not snapshot_path.is_file():
            excluded.append({"path": label, "reason": "MISSING_SIBLING_SNAPSHOT_OR_AUDIT"})
            continue
        try:
            audit = jread(audit_file)
            expected = audit.get("feature_sha256")
            if not expected:
                raise ValueError("NO_FEATURE_SHA256_IN_AUDIT")
            if sha256(snapshot_path) != expected:
                raise ValueError("FEATURE_SNAPSHOT_SHA256_MISMATCH")

            asof = utc(audit.get("asof_utc"))
            created = utc(audit.get("created_at_utc"))
            if asof is None or created is None:
                raise ValueError("MISSING_SNAPSHOT_ASOF_OR_CREATION_TIME")
            if created < asof:
                raise ValueError("CREATION_BEFORE_SNAPSHOT_ASOF")
            if audit.get("no_current_game_outcomes") is not True:
                raise ValueError("NO_EXPLICIT_BLANK_OUTCOME_ATTESTATION")
            if audit.get("v3_historical_parity") is not True:
                raise ValueError("NO_V3_HISTORICAL_FEATURE_PARITY")
            if audit.get("v45_historical_parity") is not True:
                raise ValueError("NO_V45_HISTORICAL_FEATURE_PARITY")

            forecasts = pd.read_csv(path, dtype={"game_id": "string"})
            snapshot = pd.read_parquet(snapshot_path)
            required = {"game_id", "away", "home", "kickoff_utc", "fair_margin_home", "fair_total"}
            absent = required - set(forecasts)
            if absent:
                raise ValueError("MISSING_FORECAST_COLUMNS:" + ",".join(sorted(absent)))
            for col in ("game_id", "team", "home_away"):
                if col not in snapshot:
                    raise ValueError(f"SNAPSHOT_MISSING_COLUMN:{col}")
            snapshot = snapshot.copy()
            snapshot["game_id"] = snapshot["game_id"].map(norm_id)
            forecasts = forecasts.copy()
            forecasts["game_id"] = forecasts["game_id"].map(norm_id)
            if forecasts["game_id"].duplicated().any():
                raise ValueError("DUPLICATE_FORECAST_GAME_ID")

            good, wrong = 0, 0
            for _, r in forecasts.iterrows():
                ident = str(r["game_id"])
                kickoff = utc(r["kickoff_utc"])
                game_label = f"{label}::{ident}"
                reason = None
                if kickoff is None or kickoff.year != SEASON:
                    reason = "NOT_VALID_2026_KICKOFF"
                elif not (asof < kickoff and created < kickoff):
                    reason = "SNAPSHOT_TIME_NOT_BEFORE_KICKOFF"
                elif not ident:
                    reason = "EMPTY_GAME_ID"
                else:
                    sides = snapshot.loc[snapshot["game_id"].eq(ident)]
                    if len(sides) != 2:
                        reason = "MISSING_OR_DUPLICATE_PAIR_IN_SNAPSHOT"
                    elif set(sides["home_away"].astype(str).str.lower()) != {"home", "away"}:
                        reason = "INVALID_HOME_AWAY_PAIR"
                    elif not is_blank_game_outcome(sides):
                        reason = "CURRENT_GAME_RESULT_PRESENT_IN_SNAPSHOT"
                    else:
                        h = sides.loc[sides["home_away"].astype(str).str.lower().eq("home")].iloc[0]
                        a = sides.loc[sides["home_away"].astype(str).str.lower().eq("away")].iloc[0]
                        if norm_team(h["team"]) != norm_team(r["home"]) or norm_team(a["team"]) != norm_team(r["away"]):
                            reason = "FORECAST_SNAPSHOT_TEAM_MISMATCH"
                if reason:
                    wrong += 1
                    excluded.append({"path": game_label, "reason": reason})
                    continue

                try:
                    m57 = float(r["fair_margin_home"])
                    t57 = float(r["fair_total"])
                    if not np.isfinite([m57, t57]).all() or t57 < 0:
                        raise ValueError("NONFINITE_OR_NEGATIVE_PREDICTION")
                    if "fair_home_spread" in forecasts:
                        if not np.isclose(float(r["fair_home_spread"]), -m57, atol=1e-5):
                            raise ValueError("V57_SPREAD_SIGN_INCONSISTENCY")
                    m58 = None
                    if "blend_margin_home" in forecasts and pd.notna(r["blend_margin_home"]):
                        m58 = float(r["blend_margin_home"])
                        if not np.isfinite(m58):
                            raise ValueError("NONFINITE_V58_BLEND")
                        if "blend_home_spread" in forecasts:
                            if not np.isclose(float(r["blend_home_spread"]), -m58, atol=1e-5):
                                raise ValueError("V58_SPREAD_SIGN_INCONSISTENCY")
                except (ValueError, TypeError) as ex:
                    wrong += 1
                    excluded.append({"path": game_label, "reason": str(ex)})
                    continue

                def optional_float(key):
                    if key not in forecasts or pd.isna(r[key]):
                        return None
                    try:
                        f = float(r[key])
                    except (ValueError, TypeError):
                        return None
                    return f if math.isfinite(f) else None

                records.append({
                    "game_id": ident, "away": str(r["away"]), "home": str(r["home"]),
                    "kickoff_utc": kickoff, "archived_snapshot_utc": created,
                    "archive_asof_utc": asof, "archive_path": label,
                    "v57_home_margin": m57, "v58_home_margin": m58,
                    "predicted_total": t57,
                    "margin_p10": optional_float("margin_p10"),
                    "margin_p90": optional_float("margin_p90"),
                    "total_p10": optional_float("total_p10"),
                    "total_p90": optional_float("total_p90"),
                    "snapshot_sha256": expected,
                    "historical_pit_provenance_certified": bool(audit.get("historical_pit_provenance_certified", False)),
                    "week": int(h["week_num"]) if "week_num" in snapshot and pd.notna(h["week_num"]) else None,
                })
                good += 1
            archive_report.append({"path": label, "accepted_game_rows": good, "excluded_game_rows": wrong})
        except Exception as ex:
            excluded.append({"path": label, "reason": "ARCHIVE_VALIDATION_FAILURE:" + str(ex)})

    return records, excluded, candidate_paths, archive_report


def metrics(df: pd.DataFrame) -> dict:
    out = {"n": int(len(df))}
    if not len(df):
        return out
    out["v57_margin_mae"] = float(df["v57_margin_abs_error"].mean())
    out["v57_margin_rmse"] = float(np.sqrt(np.mean(df["v57_margin_error"] ** 2)))
    out["v57_margin_bias_pred_minus_actual"] = float(df["v57_margin_error"].mean())
    out["total_mae"] = float(df["total_abs_error"].mean())
    out["total_rmse"] = float(np.sqrt(np.mean(df["total_error"] ** 2)))
    out["total_bias_pred_minus_actual"] = float(df["total_error"].mean())
    out["v57_winner_correct"] = int(df["v57_winner_correct"].sum())
    out["v57_winner_accuracy"] = float(df["v57_winner_correct"].mean())
    out["v57_margin_abs_error_p90"] = float(df["v57_margin_abs_error"].quantile(.9))
    out["total_abs_error_p90"] = float(df["total_abs_error"].quantile(.9))
    out["total_error_over_20_points"] = int(df["total_abs_error"].gt(20).sum())
    out["margin_error_over_14_points"] = int(df["v57_margin_abs_error"].gt(14).sum())
    paired = df.dropna(subset=["v58_margin_abs_error"])
    out["v58_paired_n"] = int(len(paired))
    if len(paired):
        out["paired_v57_margin_mae"] = float(paired["v57_margin_abs_error"].mean())
        out["paired_v58_margin_mae"] = float(paired["v58_margin_abs_error"].mean())
        out["paired_v58_minus_v57_margin_mae"] = float((paired["v58_margin_abs_error"] - paired["v57_margin_abs_error"]).mean())
        out["paired_v58_winner_correct"] = int(paired["v58_winner_correct"].sum())
        out["paired_v58_margin_rmse"] = float(np.sqrt(np.mean(paired["v58_margin_error"] ** 2)))
        out["paired_v58_margin_bias_pred_minus_actual"] = float(paired["v58_margin_error"].mean())
    for prefix, actual, low, high in (("margin", "actual_home_margin", "margin_p10", "margin_p90"),
                                      ("total", "actual_total", "total_p10", "total_p90")):
        band = df.dropna(subset=[low, high])
        out[f"{prefix}_p10_p90_available_n"] = int(len(band))
        if len(band):
            out[f"{prefix}_p10_p90_empirical_coverage"] = float(((band[actual] >= band[low]) & (band[actual] <= band[high])).mean())
    return out


def main():
    print("=" * 85)
    print("CFB-QUANT 2026 SEASON-TO-DATE CHRONOLOGICAL FORECAST AUDIT")
    print("Read-only prospective archive audit; no refitting/reforecasting")
    print("=" * 85)
    now = pd.Timestamp.now(tz="UTC")
    by_id, by_teams, schedule_stats = build_results_index()
    records, excluded, candidates, archive_report = scan_archives()

    all_candidates = pd.DataFrame(records)
    if not all_candidates.empty:
        # Deterministic policy: latest qualifying pregame snapshot per game.
        all_candidates = all_candidates.sort_values(
            ["game_id", "archived_snapshot_utc", "archive_path"], kind="stable"
        )
        chosen = all_candidates.drop_duplicates("game_id", keep="last").copy()
    else:
        chosen = pd.DataFrame()

    graded, unresolved, future = [], [], []
    for _, r in chosen.iterrows():
        label = f"{r['away']} @ {r['home']} ({r['game_id']})"
        if r["kickoff_utc"] >= now:
            future.append(label)
            continue
        result, problem = score_for(r, by_id, by_teams)
        if problem:
            unresolved.append({"game": label, "reason": problem, "archive_path": r["archive_path"]})
            continue
        am = float(result["actual_home"] - result["actual_away"])
        at = float(result["actual_home"] + result["actual_away"])
        pm = float(r["v57_home_margin"])
        pt = float(r["predicted_total"])
        bm = float(r["v58_home_margin"]) if pd.notna(r["v58_home_margin"]) else None
        if am == 0 or pm == 0:
            v57_win = np.nan
        else:
            v57_win = bool((pm > 0) == (am > 0))
        if bm is not None and am != 0 and bm != 0:
            v58_win = bool((bm > 0) == (am > 0))
        else:
            v58_win = np.nan
        graded.append({**r.to_dict(),
                       "actual_away": int(result["actual_away"]),
                       "actual_home": int(result["actual_home"]),
                       "actual_home_margin": am, "actual_total": at,
                       "result_source": result["result_source"],
                       "week": r["week"] if pd.notna(r["week"]) else result["week"],
                       "v57_margin_error": pm - am,
                       "v57_margin_abs_error": abs(pm - am),
                       "total_error": pt - at,
                       "total_abs_error": abs(pt - at),
                       "v57_winner_correct": v57_win,
                       "v58_margin_error": bm - am if bm is not None else np.nan,
                       "v58_margin_abs_error": abs(bm - am) if bm is not None else np.nan,
                       "v58_winner_correct": v58_win})

    df = pd.DataFrame(graded)
    if not df.empty:
        df = df.sort_values(["kickoff_utc", "game_id"], kind="stable").reset_index(drop=True)
        # Winner games ties count in graded n, but no correct/incorrect win call.
        for winner_col in ("v57_winner_correct", "v58_winner_correct"):
            df[winner_col] = pd.to_numeric(df[winner_col], errors="coerce")
        df["pred_total_band"] = pd.cut(
            df["predicted_total"],
            bins=[-np.inf, 44.999999, 54.999999, 64.999999, 74.999999, np.inf],
            labels=["<45", "45-54.9", "55-64.9", "65-74.9", "75+"],
            right=True,
        )

    eligible_meta = {
        "audit_date_utc": now.isoformat(), "season": SEASON,
        "model_training_reexecuted": False,
        "original_forecast_or_snapshot_modified": False,
        "forecast_selection": "latest eligible archived snapshot before kickoff, ties by archive path",
        "method": "score only hashed archived paired_forecasts.csv with sibling pregame_features.parquet, audit.json, verified blank game outcomes and 2026 results",
        "time_limit": "archive snapshot creation and asof must precede game kickoff; no proof forecast CSV itself was immutable before kickoff",
        "historical_pit_provenance_certified": False,
        "n_archive_csv_candidates": len(candidates),
        "n_verified_pregame_forecast_rows_in_inventory": int(len(records)),
        "n_unique_qualifying_games": int(len(chosen)),
        "n_graded": int(len(df)),
        "n_unresolved_scores": len(unresolved),
        "n_not_started": len(future),
        "n_excluded_archive_or_game_records": len(excluded),
        "local_schedule_stats": dict(schedule_stats),
        "metrics": metrics(df),
    }
    if len(df):
        eligible_meta["earliest_graded_kickoff_utc"] = df.iloc[0]["kickoff_utc"].isoformat()
        eligible_meta["latest_graded_kickoff_utc"] = df.iloc[-1]["kickoff_utc"].isoformat()
        eligible_meta["n_graded_prior_to_2026_10_20"] = int((df["kickoff_utc"] < pd.Timestamp("2026-10-20", tz="UTC")).sum())
    else:
        eligible_meta["n_graded_prior_to_2026_10_20"] = 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPORTS / "v58_development" / "chronological_2026_audit" / stamp
    out.mkdir(parents=True, exist_ok=False)
    if len(df):
        df.to_csv(out / "graded_forecasts.csv", index=False)
        # Group scorecards include game count, error, bias, winner accuracy.
        if df["week"].notna().any():
            weeks = []
            for week, group in df.groupby("week", dropna=False):
                weeks.append({"week": week, **metrics(group)})
            pd.DataFrame(weeks).to_csv(out / "by_week.csv", index=False)
        bands = []
        for band, group in df.groupby("pred_total_band", observed=True):
            bands.append({"projected_total_band": str(band), **metrics(group)})
        pd.DataFrame(bands).to_csv(out / "by_predicted_total.csv", index=False)
        extremes = df.sort_values("total_abs_error", ascending=False).head(20)
        extremes.to_csv(out / "largest_total_misses.csv", index=False)
        extremes = df.sort_values("v57_margin_abs_error", ascending=False).head(20)
        extremes.to_csv(out / "largest_margin_misses.csv", index=False)
    pd.DataFrame(excluded, columns=["path", "reason"]).to_csv(out / "excluded.csv", index=False)
    pd.DataFrame(unresolved, columns=["game", "reason", "archive_path"]).to_csv(out / "missing_scores.csv", index=False)
    pd.DataFrame(archive_report, columns=["path", "accepted_game_rows", "excluded_game_rows"]).to_csv(out / "archive_inventory.csv", index=False)
    pd.DataFrame({"candidate_file": [str(p.relative_to(ROOT)) for p in candidates]}).to_csv(out / "all_candidate_csvs.csv", index=False)
    (out / "summary.json").write_text(json.dumps(eligible_meta, indent=2, default=str), encoding="utf-8")

    print(f"\nForecast CSV candidates discovered: {len(candidates)}")
    print(f"Verified pregame archived forecast rows: {len(records)}")
    print(f"Unique qualifying games: {len(chosen)}")
    print(f"Completed games with verified/available scores: {len(df)}")
    print(f"Finished games without scores: {len(unresolved)}")
    print(f"Future games not graded: {len(future)}")
    print(f"Rejected archive/game records: {len(excluded)}")
    print("\nOVERALL ACCURACY (margin bias/total bias = prediction minus actual):")
    print(json.dumps(eligible_meta["metrics"], indent=2))
    if len(df):
        print("\nCHRONOLOGICAL GAME GRADES:")
        cols = ["kickoff_utc", "away", "home", "actual_away", "actual_home",
                "v57_home_margin", "v58_home_margin", "predicted_total",
                "v57_margin_abs_error", "total_abs_error"]
        print(df[cols].round(3).to_string(index=False, max_rows=60))
        print("\nLARGEST TOTAL MISSES:")
        print(df.nlargest(10, "total_abs_error")[["away", "home", "actual_total", "predicted_total", "total_abs_error"]].round(2).to_string(index=False))
        print("\nPER-WEEK SCORECARDS:")
        if (out / "by_week.csv").exists():
            print(pd.read_csv(out / "by_week.csv").round(3).to_string(index=False))
        else:
            print("No verified week identifiers")
        print("\nBY PREDICTED-TOTAL BAND:")
        print(pd.read_csv(out / "by_predicted_total.csv").round(3).to_string(index=False))
    if unresolved:
        print("\nMISSING FINALS (first 15):")
        for u in unresolved[:15]:
            print("-", u["game"], "=>", u["reason"])
    if excluded:
        print("\nEXCLUDED FORECASTS (first 15):")
        for e in excluded[:15]:
            print("-", e["path"], "=>", e["reason"])
    print("\nCOVERAGE WARNING:")
    if eligible_meta["n_graded_prior_to_2026_10_20"] == 0:
        print("NO VERIFIED EARLIER 2026 PREGAME FORECASTS GRADED. This is not a complete season-to-date audit.")
    else:
        print("Earlier-season archived forecasts exist; audit covers ONLY discoverable, validated archives, not all 2026 games.")
    print("Historical point-in-time source certification: NOT ESTABLISHED.")
    print("Archive creation times do not independently certify forecast CSV immutability.")
    print("No original forecasts, training files, or source schedules changed.")
    print("\nREPORT DIRECTORY:", out)
    print("=" * 85)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"AUDIT FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

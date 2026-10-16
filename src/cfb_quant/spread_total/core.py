"""Chronological paired-score residual modelling and push-aware market math.

The bootstrap is a research baseline, NOT a validated production probability
model. It retains home/away residual dependence without assuming normal tails.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


HISTORY_REQUIRED = {
    "game_id", "season", "week", "team", "home_away", "target", "model",
    "prediction", "actual", "train_end_season",
}


def audit_game_pairs(oof: pd.DataFrame, model: str = "ridge") -> pd.DataFrame:
    """Describe home/away coverage before deciding which games are usable.

    One-sided OOF history is expected when the source excludes an opponent or
    the eligibility mask excludes a team (e.g. no prior games).  We report it
    explicitly; we never synthesize a missing team's prediction.
    """
    missing = HISTORY_REQUIRED.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF file is missing columns: {sorted(missing)}")
    s = oof.loc[(oof["target"] == "points") & (oof["model"] == model)].copy()
    if s.empty:
        raise ValueError(f"No points predictions found for model={model!r}")
    for col in ("prediction", "actual", "season", "week", "train_end_season", "game_id"):
        s[col] = pd.to_numeric(s[col], errors="coerce")
    if s[["prediction", "actual", "season", "week", "train_end_season", "game_id"]].isna().any().any():
        raise ValueError("Invalid numeric values in points OOF history")
    if not np.isfinite(s[["prediction", "actual"]].to_numpy(dtype=float)).all():
        raise ValueError("Infinite points values in OOF history")
    if (s["train_end_season"] >= s["season"]).any():
        raise ValueError("Chronology violation: a forecast trained on its prediction season or later")
    if not s["home_away"].isin(["home", "away"]).all():
        raise ValueError("home_away must contain only 'home' or 'away'")
    if s.duplicated(["game_id", "home_away"]).any():
        raise ValueError("Duplicate home/away predictions for a game")
    meta = s.groupby("game_id").agg(
        season=("season", "first"), week=("week", "first"),
        n_rows=("home_away", "size"), n_seasons=("season", "nunique"),
        n_weeks=("week", "nunique"), n_teams=("team", "nunique"),
    )
    by_side = pd.crosstab(s["game_id"], s["home_away"]).reindex(
        index=meta.index, columns=["home", "away"], fill_value=0)
    audit = meta.join(by_side.rename(columns={"home": "n_home", "away": "n_away"}))
    audit["status"] = np.select(
        [
            (audit.n_rows.eq(2) & audit.n_home.eq(1) & audit.n_away.eq(1)
             & audit.n_seasons.eq(1) & audit.n_weeks.eq(1) & audit.n_teams.eq(2)),
            audit.n_rows.eq(1) & audit.n_home.eq(1),
            audit.n_rows.eq(1) & audit.n_away.eq(1),
        ], ["complete", "missing_away", "missing_home"], default="invalid")
    return audit.reset_index().sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def build_game_errors(oof: pd.DataFrame, model: str = "ridge", *,
                      allow_incomplete: bool = False) -> pd.DataFrame:
    """Pair OOF scores.  Incomplete games may only be skipped explicitly.

    Duplicate and contradictory metadata always fail.  Only two-sided games
    are used to estimate joint home/away scoring distributions.
    """
    audit = audit_game_pairs(oof, model=model)
    invalid = audit[audit.status.eq("invalid")]
    if not invalid.empty:
        raise ValueError(f"Invalid game pairing metadata; examples: "
                         f"{invalid[['game_id', 'season', 'week']].head(5).to_dict('records')}")
    incomplete = audit[~audit.status.eq("complete")]
    if len(incomplete) and not allow_incomplete:
        raise ValueError("Every OOF game must have exactly one home and one away prediction. "
                         f"{len(incomplete)} incomplete games; "
                         "rerun with allow_incomplete=True only after reviewing pair audit")
    complete_ids = set(audit.loc[audit.status.eq("complete"), "game_id"])
    if not complete_ids:
        raise ValueError("No complete two-sided OOF games available")
    s = oof.loc[(oof["target"] == "points") & (oof["model"] == model) &
                (pd.to_numeric(oof["game_id"], errors="coerce").isin(complete_ids))].copy()
    for col in ("game_id", "season", "week", "prediction", "actual", "train_end_season"):
        s[col] = pd.to_numeric(s[col], errors="raise")
    home = s.loc[s.home_away == "home", [
        "game_id", "season", "week", "team", "prediction", "actual", "train_end_season",
    ]].rename(columns={"team":"home_team", "prediction":"home_pred", "actual":"home_actual", "train_end_season":"home_train_end"})
    away = s.loc[s.home_away == "away", [
        "game_id", "season", "week", "team", "prediction", "actual", "train_end_season",
    ]].rename(columns={"team":"away_team", "prediction":"away_pred", "actual":"away_actual", "train_end_season":"away_train_end"})
    out = home.merge(away, on=["game_id", "season", "week"], how="inner", validate="one_to_one")
    if len(out) != len(complete_ids):
        raise ValueError("Home/away pairing failed, or home/away metadata disagree")
    out["home_error"] = out.home_actual - out.home_pred
    out["away_error"] = out.away_actual - out.away_pred
    out["actual_total"] = out.home_actual + out.away_actual
    out["actual_margin"] = out.home_actual - out.away_actual
    out["pred_total"] = out.home_pred + out.away_pred
    out["pred_margin"] = out.home_pred - out.away_pred
    return out.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def prior_errors(games: pd.DataFrame, *, before_season: int,
                 min_games: int = 300, min_seasons: int = 3) -> pd.DataFrame:
    """Only earlier *seasons* are eligible, no current-season outcome leakage."""
    prior = games.loc[pd.to_numeric(games.season).lt(before_season)].copy()
    if len(prior) < min_games or prior.season.nunique() < min_seasons:
        raise ValueError(
            f"Insufficient earlier-season OOF residuals: {len(prior)} games, "
            f"{prior.season.nunique()} seasons; need {min_games} games / {min_seasons} seasons"
        )
    return prior


def stable_seed(game_id: str | int, seed: int = 731) -> int:
    digest = hashlib.sha256(f"{seed}:{game_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True)
class ScoreDistribution:
    home: np.ndarray
    away: np.ndarray

    @property
    def margin(self) -> np.ndarray:
        return self.home - self.away

    @property
    def total(self) -> np.ndarray:
        return self.home + self.away


def simulate_scores(home_pred: float, away_pred: float, prior: pd.DataFrame,
                    *, n: int = 12000, seed: int = 731) -> ScoreDistribution:
    """Joint residual pairs, rounded to integer scores and floored at zero.

    Current approach is intentionally simple so it can be evaluated against
    more sophisticated possession and heteroskedastic alternatives later.
    """
    if n < 100:
        raise ValueError("n must be >= 100")
    if not np.isfinite([home_pred, away_pred]).all():
        raise ValueError("Point predictions must be finite")
    residuals = prior[["home_error", "away_error"]].to_numpy(dtype=float)
    if len(residuals) == 0 or not np.isfinite(residuals).all():
        raise ValueError("Residual pool is empty or not finite")
    rng = np.random.default_rng(seed)
    draw = residuals[rng.integers(0, len(residuals), size=n)]
    home = np.maximum(0, np.rint(home_pred + draw[:, 0])).astype(int)
    away = np.maximum(0, np.rint(away_pred + draw[:, 1])).astype(int)
    return ScoreDistribution(home=home, away=away)


def fair_summary(dist: ScoreDistribution) -> dict[str, float]:
    t, m = dist.total, dist.margin
    return {
        "fair_total_mean": float(t.mean()),
        "fair_total_median": float(np.median(t)),
        "total_p10": float(np.quantile(t, .10)),
        "total_p90": float(np.quantile(t, .90)),
        "fair_home_margin_mean": float(m.mean()),
        "fair_home_spread": float(-m.mean()),
        "margin_p10": float(np.quantile(m, .10)),
        "margin_p90": float(np.quantile(m, .90)),
        "p_home_win": float(np.mean(m > 0)),
        "p_tie_regulation_score": float(np.mean(m == 0)),
    }


def settle(dist: ScoreDistribution, *, market: str,
           selection: str, line: float) -> tuple[float, float, float]:
    """Returns (p_win, p_push, p_lose). Spread line is for the SELECTED side.

    NOTE: Model distribution uses final scores (including overtime if present in
    source historical results); observed tied scores have their own consequences.
    """
    if not np.isfinite(line):
        raise ValueError("line must be finite")
    market = market.lower().strip()
    selection = selection.lower().strip()
    if market == "spread" and selection in ("home", "away"):
        result = (dist.margin if selection == "home" else -dist.margin) + line
    elif market == "total" and selection in ("over", "under"):
        result = (dist.total - line) if selection == "over" else (line - dist.total)
    else:
        raise ValueError("Expected spread/home|away or total/over|under")
    return (float(np.mean(result > 0)), float(np.mean(result == 0)),
            float(np.mean(result < 0)))


def decimal_profit_per_unit(american_odds: int) -> float:
    if american_odds == 0 or abs(american_odds) < 100:
        raise ValueError("Valid American odds require <=-100 or >=+100")
    return 100.0 / abs(american_odds) if american_odds < 0 else american_odds / 100.0


def expected_roi(p_win: float, p_push: float, p_lose: float,
                 american_odds: int) -> float:
    if not np.isclose(p_win + p_push + p_lose, 1.0, atol=1e-6):
        raise ValueError("Market probabilities must sum to one")
    if min(p_win, p_push, p_lose) < 0 or max(p_win, p_push, p_lose) > 1:
        raise ValueError("Invalid probabilities")
    return p_win * decimal_profit_per_unit(american_odds) - p_lose


def empirical_crps(draws: np.ndarray, outcome: float) -> float:
    """Exact empirical CRPS in O(n log n) using sorted simulation samples."""
    x = np.sort(np.asarray(draws, dtype=float))
    n = len(x)
    if n == 0:
        raise ValueError("No samples supplied")
    mean_abs = float(np.abs(x - outcome).mean())
    pair_penalty = float(np.dot(2 * np.arange(n) - n + 1, x) / (n * n))
    return mean_abs - pair_penalty

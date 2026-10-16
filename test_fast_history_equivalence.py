import importlib.util
import importlib.machinery
import numpy as np
import pandas as pd

from cfb_quant.features import engine as fast_engine

backup_path = "src/cfb_quant/features/engine.py.pre_fast_history_backup"

loader = importlib.machinery.SourceFileLoader(
    "slow_engine",
    backup_path,
)
spec = importlib.util.spec_from_loader(
    "slow_engine",
    loader,
)
slow_engine = importlib.util.module_from_spec(spec)
loader.exec_module(slow_engine)

source = pd.read_parquet(
    "data/processed/canonical_player_games_2026.parquet"
).copy()

source = slow_engine._sort_frame(source)

sample = (
    source.groupby("_player_key", sort=False)
    .head(8)
    .head(5000)
    .copy()
)

metrics = [
    "pass_attempts",
    "completions",
    "passing_yards",
    "rush_attempts",
    "rushing_yards",
    "receptions",
    "receiving_yards",
]

slow_out, slow_cols = slow_engine._history_features(
    sample,
    group_col="_player_key",
    metrics=metrics,
    prefix="test_",
    windows=(3, 5),
    include_std5=True,
)

fast_out, fast_cols = fast_engine._history_features(
    sample,
    group_col="_player_key",
    metrics=metrics,
    prefix="test_",
    windows=(3, 5),
    include_std5=True,
)

print("=" * 90)
print("CFB FAST HISTORY EQUIVALENCE TEST")
print("=" * 90)

print("Rows:", len(sample))
print("Slow feature count:", len(slow_cols))
print("Fast feature count:", len(fast_cols))
print("Column lists identical:", slow_cols == fast_cols)

failures = []

for col in slow_cols:
    a = pd.to_numeric(
        slow_out[col],
        errors="coerce",
    ).to_numpy(dtype=float)

    b = pd.to_numeric(
        fast_out[col],
        errors="coerce",
    ).to_numpy(dtype=float)

    same = np.allclose(
        a,
        b,
        rtol=1e-10,
        atol=1e-12,
        equal_nan=True,
    )

    if not same:
        diff = np.nanmax(
            np.abs(a - b)
        )
        failures.append(
            (col, diff)
        )

print(
    "Mismatched columns:",
    len(failures),
)

if failures:
    print("\nFIRST MISMATCHES")
    for col, diff in failures[:20]:
        print(col, diff)
else:
    print(
        "\nPASS: FAST IMPLEMENTATION MATCHES ORIGINAL."
    )

print("=" * 90)

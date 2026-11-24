
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "multi_race"
    / "runs"
    / "20261124_175647"
    / "multi_race_integrated.csv"
)

df = pd.read_csv(DATA_PATH)

df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")

valid = (
    df["is_valid_racing_lap"]
    .astype(str)
    .str.lower()
    .eq("true")
)

strict_dry = df.loc[
    valid
    & df["event_name"].eq("British Grand Prix")
    & df["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
    & df["LapNumber"].between(2, 16)
].copy()

# Composite stint identity
stint_keys = ["session_id", "Driver", "Stint"]

print("=" * 100)
print("BRITISH GP STRICT-DRY STINT SUPPORT")
print("=" * 100)

print(f"Observations: {len(strict_dry)}")
print(f"Drivers: {strict_dry['Driver'].nunique()}")

stints = (
    strict_dry.groupby(stint_keys)
    .agg(
        Compound=("Compound", "first"),
        observations=("LapNumber", "size"),
        first_lap=("LapNumber", "min"),
        last_lap=("LapNumber", "max"),
        first_age=("TyreLife", "min"),
        last_age=("TyreLife", "max"),
        unique_ages=("TyreLife", "nunique"),
        age_std=("TyreLife", "std"),
    )
    .reset_index()
)

stints["age_span"] = stints["last_age"] - stints["first_age"]

print("\n1. INDIVIDUAL STINT SUPPORT")
print(stints.sort_values(["Compound", "Driver", "first_lap"]).to_string(index=False))

print("\n2. STINT SUPPORT BY COMPOUND")
print(
    stints.groupby("Compound")
    .agg(
        stints=("observations", "size"),
        observations=("observations", "sum"),
        median_laps_per_stint=("observations", "median"),
        minimum_laps_per_stint=("observations", "min"),
        maximum_laps_per_stint=("observations", "max"),
        median_age_span=("age_span", "median"),
        zero_age_variation=("age_span", lambda x: (x == 0).sum()),
    )
    .to_string()
)

print("\n3. STINT LENGTH DISTRIBUTION")
print(stints["observations"].value_counts().sort_index().to_string())

print("\n4. TYRE AGE SUPPORT BY COMPOUND")
print(
    strict_dry.groupby("Compound")["TyreLife"]
    .agg(["count", "min", "max", "mean", "std", "nunique"])
    .round(2)
    .to_string()
)

print("\n5. OBSERVATION-LEVEL AGE COVERAGE")
print(
    strict_dry.groupby(["Compound", "TyreLife"])
    .size()
    .unstack(fill_value=0)
    .to_string()
)

print("\nSTRICT-DRY STINT SUPPORT AUDIT COMPLETE")

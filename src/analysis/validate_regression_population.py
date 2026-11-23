
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "processed"
    / "2024_british_gp_fia_integrated.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed"
    / "regression_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

def to_bool(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )

for col in [
    "is_valid_racing_lap",
    "is_pit_transition",
    "is_deleted",
]:
    df[col] = to_bool(df[col])

df["LapNumber"] = pd.to_numeric(
    df["LapNumber"], errors="coerce"
)

df["lap_time_s"] = pd.to_numeric(
    df["lap_time_s"], errors="coerce"
)

df["TyreLife"] = pd.to_numeric(
    df["TyreLife"], errors="coerce"
)

# Reconstruct the A1 Strict Slick population
valid = df[
    df["is_valid_racing_lap"]
    & df["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
    & df["lap_time_s"].notna()
    & (df["lap_time_s"] > 0)
    & df["TyreLife"].notna()
].copy()

# Intermediate population
intermediate = df[
    df["is_valid_racing_lap"]
    & (df["Compound"] == "INTERMEDIATE")
    & df["lap_time_s"].notna()
    & (df["lap_time_s"] > 0)
].copy()

# Focus on key laps
laps = [18, 19, 20, 25, 26, 27, 28, 29, 30]

rows = []

for lap in laps:
    original = df[df["LapNumber"] == lap]
    valid_lap = valid[valid["LapNumber"] == lap]
    inter_lap = intermediate[
        intermediate["LapNumber"] == lap
    ]

    rows.append({
        "LapNumber": lap,
        "original_records": len(original),
        "valid_racing_records": int(
            original["is_valid_racing_lap"].sum()
        ),
        "slick_records": len(valid_lap),
        "intermediate_records": len(inter_lap),
        "pit_transition_records": int(
            original["is_pit_transition"].sum()
        ),
        "deleted_records": int(
            original["is_deleted"].sum()
        ),
        "slick_unique_drivers": valid_lap["Driver"].nunique(),
        "intermediate_unique_drivers": (
            inter_lap["Driver"].nunique()
        ),
    })

result = pd.DataFrame(rows)

print("\nREGRESSION POPULATION VALIDATION")
print("=" * 110)

print(result.to_string(index=False))

result.to_csv(
    OUTPUT_DIR / "regression_population_validation.csv",
    index=False
)

print("\nTOTAL POPULATION COUNTS")
print("=" * 90)

print("Original records:", len(df))
print("Valid slick records:", len(valid))
print("Valid intermediate records:", len(intermediate))
print("Pit-transition records:", int(df["is_pit_transition"].sum()))
print("Officially deleted records:", int(df["is_deleted"].sum()))

print("\nExported:")
print(OUTPUT_DIR / "regression_population_validation.csv")

print("\nPopulation validation completed.")

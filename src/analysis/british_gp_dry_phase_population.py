
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

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

BRITISH_GP = "British Grand Prix"
SLICK_COMPOUNDS = ["HARD", "MEDIUM", "SOFT"]

# Race laps 2-16: before recorded rainfall begins.
STRICT_DRY_LAPS = range(2, 17)

# Laps 17-39: rainfall onset, wet running and return-to-slick
# transition. Used for diagnostic classification only.
TRANSITION_LAPS = range(17, 40)


# ---------------------------------------------------------
# LOAD AND VALIDATE
# ---------------------------------------------------------

print("Loading multi-race integrated dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

required_columns = [
    "session_id",
    "event_name",
    "Driver",
    "Stint",
    "Compound",
    "LapNumber",
    "TyreLife",
    "lap_time_s",
    "Rainfall",
    "is_valid_racing_lap",
]

missing_columns = [c for c in required_columns if c not in df.columns]

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

df["Rainfall"] = (
    df["Rainfall"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"true": True, "false": False})
)

df["LapNumber"] = pd.to_numeric(df["LapNumber"], errors="coerce")
df["TyreLife"] = pd.to_numeric(df["TyreLife"], errors="coerce")

valid_flag = (
    df["is_valid_racing_lap"]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)

slick_flag = df["Compound"].isin(SLICK_COMPOUNDS)

model_population = df.loc[valid_flag & slick_flag].copy()

british = model_population.loc[
    model_population["event_name"].eq(BRITISH_GP)
].copy()

if british.empty:
    raise ValueError("No valid British GP slick observations found.")

print(f"Full dataset records: {len(df):,}")
print(f"Full valid dry-slick model population: {len(model_population):,}")
print(f"British GP valid slick observations: {len(british):,}")


# ---------------------------------------------------------
# DEFINE THREE BRITISH GP POPULATIONS
# ---------------------------------------------------------

population_A = british.copy()

population_B = british.loc[
    british["LapNumber"].isin(STRICT_DRY_LAPS)
].copy()

population_C = british.loc[
    british["Rainfall"].eq(False)
].copy()

populations = {
    "A - Original British GP slick population": population_A,
    "B - Strict early dry (laps 2-16)": population_B,
    "C - Rainfall=False (whole race)": population_C,
}


# ---------------------------------------------------------
# REPORTING FUNCTION
# ---------------------------------------------------------

def report_population(name, data):
    print("\n" + "=" * 100)
    print(name)
    print("=" * 100)

    print(f"Observations: {len(data):,}")
    print(f"Drivers: {data['Driver'].nunique()}")
    print(
        "Composite stints:",
        data[["session_id", "Driver", "Stint"]]
        .drop_duplicates()
        .shape[0],
    )

    print("\nCompound distribution:")
    print(data["Compound"].value_counts().sort_index().to_string())

    print("\nTyreLife distribution:")
    print(
        data["TyreLife"]
        .describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90])
        .round(2)
        .to_string()
    )

    print("\nLapNumber range:")
    print(
        f"Min: {data['LapNumber'].min():.0f}"
        f" | Max: {data['LapNumber'].max():.0f}"
    )

    transition = data["LapNumber"].between(
        min(TRANSITION_LAPS),
        max(TRANSITION_LAPS),
    )

    print(f"\nObservations in laps 17-39: {transition.sum():,}")
    print(f"Rainfall=True observations: {data['Rainfall'].eq(True).sum():,}")
    print(f"Rainfall=False observations: {data['Rainfall'].eq(False).sum():,}")
    print(f"Missing rainfall observations: {data['Rainfall'].isna().sum():,}")

    print("\nRace-lap distribution:")
    print(data["LapNumber"].value_counts().sort_index().to_string())


# ---------------------------------------------------------
# BRITISH GP COMPARISON
# ---------------------------------------------------------

for name, population in populations.items():
    report_population(name, population)


# ---------------------------------------------------------
# FIVE-RACE CALIBRATION POPULATION COMPARISON
# ---------------------------------------------------------

print("\n" + "=" * 100)
print("FIVE-RACE POPULATION COMPARISON")
print("=" * 100)

other_races = model_population.loc[
    ~model_population["event_name"].eq(BRITISH_GP)
].copy()

print(f"Other four races - unchanged: {len(other_races):,}")

for name, british_population in populations.items():
    combined = pd.concat(
        [other_races, british_population],
        ignore_index=True,
    )

    print(f"\n{name}")
    print(f"Total five-race observations: {len(combined):,}")
    print(f"Total drivers: {combined['Driver'].nunique()}")
    print(f"Total sessions: {combined['session_id'].nunique()}")
    print(
        "Total composite stints:",
        combined[["session_id", "Driver", "Stint"]]
        .drop_duplicates()
        .shape[0],
    )

    print("\nRace distribution:")
    print(
        combined.groupby("event_name")
        .size()
        .sort_index()
        .to_string()
    )

    print("\nCompound distribution:")
    print(
        combined["Compound"]
        .value_counts()
        .sort_index()
        .to_string()
    )


print("\n" + "=" * 100)
print("BRITISH GP DRY-PHASE POPULATION AUDIT COMPLETE")
print("=" * 100)

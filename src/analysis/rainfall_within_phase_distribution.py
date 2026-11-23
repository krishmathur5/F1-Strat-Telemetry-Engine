
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    ROOT / "data" / "processed" /
    "2024_british_gp_fia_integrated.csv"
)

OUTPUT_DIR = (
    ROOT / "data" / "processed" /
    "regression_diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAP_OUTPUT = OUTPUT_DIR / "rainfall_by_lap.csv"
DRIVER_OUTPUT = OUTPUT_DIR / "post_transition_rainfall_by_driver.csv"


# ============================================================
# LOAD AND PREPARE
# ============================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(DATA_PATH)


def to_bool(series):
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


for col in ["is_valid_racing_lap", "Rainfall"]:
    if col in df.columns:
        df[col] = to_bool(df[col])

if "rain_observed" not in df.columns:
    df["rain_observed"] = df["Rainfall"]

df["rain_observed"] = to_bool(df["rain_observed"])

for col in ["LapNumber", "lap_time_s", "TyreLife"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Compound"] = df["Compound"].astype(str).str.upper()
df["Driver"] = df["Driver"].astype(str)


# ============================================================
# RECONSTRUCT A1 STRICT SLICK POPULATION
# ============================================================

df = df[
    df["is_valid_racing_lap"]
    & df["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
    & (df["lap_time_s"] > 0)
].copy()

df = df.dropna(
    subset=[
        "LapNumber",
        "lap_time_s",
        "TyreLife",
        "Driver",
        "Compound",
        "rain_observed",
    ]
).copy()

if len(df) != 658:
    raise ValueError(
        f"Expected 658 observations, found {len(df)}"
    )


# ============================================================
# RAINFALL COUNTS BY LAP
# ============================================================

lap_summary = (
    df.groupby("LapNumber")
    .agg(
        observations=("rain_observed", "size"),
        wet_observations=("rain_observed", "sum"),
        unique_drivers=("Driver", "nunique"),
    )
    .reset_index()
)

lap_summary["dry_observations"] = (
    lap_summary["observations"]
    - lap_summary["wet_observations"]
)

lap_summary["rainfall_share"] = (
    lap_summary["wet_observations"]
    / lap_summary["observations"]
)

lap_summary["rainfall_varies"] = (
    (lap_summary["wet_observations"] > 0)
    & (lap_summary["dry_observations"] > 0)
)

lap_summary.to_csv(LAP_OUTPUT, index=False)


# ============================================================
# POST-TRANSITION DRIVER COVERAGE
# ============================================================

post = df[df["LapNumber"] >= 28].copy()

driver_summary = (
    post.groupby(["Driver", "rain_observed"])
    .size()
    .unstack(fill_value=0)
)

for value in [False, True]:
    if value not in driver_summary.columns:
        driver_summary[value] = 0

driver_summary = driver_summary.rename(columns={
    False: "dry_observations",
    True: "wet_observations",
})

driver_summary["total_observations"] = (
    driver_summary["dry_observations"]
    + driver_summary["wet_observations"]
)

driver_summary["wet_share"] = (
    driver_summary["wet_observations"]
    / driver_summary["total_observations"]
)

driver_summary["has_both_conditions"] = (
    (driver_summary["dry_observations"] > 0)
    & (driver_summary["wet_observations"] > 0)
)

driver_summary = driver_summary.reset_index()

driver_summary.to_csv(DRIVER_OUTPUT, index=False)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\nRAINFALL DISTRIBUTION BY LAP")
print("=" * 100)

print(
    lap_summary[
        lap_summary["LapNumber"] >= 15
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nPOST-TRANSITION DRIVER COVERAGE")
print("=" * 100)

print(
    driver_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nPOST-TRANSITION TOTALS")
print("=" * 80)

print(f"Observations: {len(post)}")
print(f"Dry observations: {(~post['rain_observed']).sum()}")
print(f"Wet observations: {post['rain_observed'].sum()}")
print(
    "Drivers with both conditions: "
    f"{driver_summary['has_both_conditions'].sum()}"
)

print("\nEXPORTED FILES")
print(LAP_OUTPUT)
print(DRIVER_OUTPUT)

print("\nWithin-phase rainfall distribution completed.")

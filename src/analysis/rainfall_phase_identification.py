
from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATHS
# ============================================================

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

PHASE_SUMMARY_PATH = (
    OUTPUT_DIR / "rainfall_phase_identification_summary.csv"
)

DRIVER_DETAIL_PATH = (
    OUTPUT_DIR / "rainfall_phase_driver_detail.csv"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original records: {len(df)}")


# ============================================================
# 3. NORMALISE DATA TYPES
# ============================================================

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

for col in ["lap_time_s", "TyreLife", "LapNumber", "TrackTemp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Compound"] = df["Compound"].astype(str).str.upper()
df["Driver"] = df["Driver"].astype(str)


# ============================================================
# 4. RECONSTRUCT A1 STRICT SLICK POPULATION
# ============================================================

df = df[
    df["is_valid_racing_lap"]
    & df["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
    & (df["lap_time_s"] > 0)
].copy()

required = [
    "lap_time_s",
    "rain_observed",
    "TrackTemp",
    "TyreLife",
    "LapNumber",
    "Driver",
    "Compound",
]

df = df.dropna(subset=required).copy()

print(f"A1 Strict Slick observations: {len(df)}")

if len(df) != 658:
    raise ValueError(
        f"Expected 658 observations, found {len(df)}. "
        "Stop and investigate population filtering."
    )


# ============================================================
# 5. ASSIGN RACE PHASES
# ============================================================

def classify_phase(lap):
    if lap <= 17:
        return "Pre-transition"
    elif lap <= 20:
        return "Initial wet transition"
    elif lap <= 25:
        return "Wet running"
    elif lap <= 27:
        return "Late wet / pit phase"
    else:
        return "Post-transition running"


phase_order = [
    "Pre-transition",
    "Initial wet transition",
    "Wet running",
    "Late wet / pit phase",
    "Post-transition running",
]

df["race_phase"] = pd.Categorical(
    df["LapNumber"].apply(classify_phase),
    categories=phase_order,
    ordered=True,
)


# ============================================================
# 6. PHASE-LEVEL RAINFALL COUNTS
# ============================================================

phase_rows = []

for phase in phase_order:
    subset = df[df["race_phase"] == phase]

    dry_count = int((~subset["rain_observed"]).sum())
    wet_count = int(subset["rain_observed"].sum())

    driver_counts = (
        subset.groupby(["Driver", "rain_observed"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    # Ensure both Boolean columns exist.
    for value in [False, True]:
        if value not in driver_counts.columns:
            driver_counts[value] = 0

    drivers_with_both = int(
        (
            (driver_counts[False] > 0)
            & (driver_counts[True] > 0)
        ).sum()
    )

    phase_rows.append({
        "race_phase": phase,
        "observations": len(subset),
        "dry_observations": dry_count,
        "wet_observations": wet_count,
        "rainfall_share": (
            wet_count / len(subset) if len(subset) else None
        ),
        "unique_drivers": subset["Driver"].nunique(),
        "drivers_with_both_conditions": drivers_with_both,
        "first_lap": (
            int(subset["LapNumber"].min())
            if len(subset) else None
        ),
        "last_lap": (
            int(subset["LapNumber"].max())
            if len(subset) else None
        ),
        "rainfall_varies_within_phase": (
            dry_count > 0 and wet_count > 0
        ),
    })

phase_summary = pd.DataFrame(phase_rows)

phase_summary.to_csv(
    PHASE_SUMMARY_PATH,
    index=False
)


# ============================================================
# 7. DRIVER-LEVEL RAINFALL COVERAGE WITHIN EACH PHASE
# ============================================================

driver_detail = (
    df.groupby(
        ["race_phase", "Driver", "rain_observed"],
        observed=True
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

# Ensure dry and wet columns both exist.
for value in [False, True]:
    if value not in driver_detail.columns:
        driver_detail[value] = 0

driver_detail = driver_detail.rename(columns={
    False: "dry_observations",
    True: "wet_observations",
})

driver_detail["has_both_conditions"] = (
    (driver_detail["dry_observations"] > 0)
    & (driver_detail["wet_observations"] > 0)
)

driver_detail.to_csv(
    DRIVER_DETAIL_PATH,
    index=False
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\nRAINFALL VARIATION BY RACE PHASE")
print("=" * 115)

print(
    phase_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nDRIVER-LEVEL COVERAGE BY PHASE")
print("=" * 115)

for phase in phase_order:
    subset = driver_detail[
        driver_detail["race_phase"] == phase
    ]

    both = int(subset["has_both_conditions"].sum())

    print(
        f"{phase}: "
        f"{len(subset)} drivers; "
        f"{both} drivers observed in both dry and wet conditions"
    )

print("\nEXPORTED FILES")
print("=" * 80)
print(PHASE_SUMMARY_PATH)
print(DRIVER_DETAIL_PATH)

print("\nRainfall phase identification completed.")

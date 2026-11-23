
from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt


# ============================================================
# 1. PATHS AND CONFIGURATION
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

SUMMARY_PATH = OUTPUT_DIR / "race_phase_residual_summary.csv"
LAP_PATH = OUTPUT_DIR / "race_phase_residual_by_lap.csv"
OBSERVATIONS_PATH = OUTPUT_DIR / "race_phase_residual_observations.csv"
PLOT_PATH = OUTPUT_DIR / "race_phase_residual_analysis.png"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Original records: {len(df)}")


# ============================================================
# 3. NORMALISE BOOLEAN COLUMNS
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


# ============================================================
# 4. RECONSTRUCT A1 STRICT SLICK POPULATION
# ============================================================

numeric_columns = [
    "lap_time_s",
    "TyreLife",
    "LapNumber",
    "TrackTemp",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

required_columns = [
    "is_valid_racing_lap",
    "rain_observed",
    "TrackTemp",
    "TyreLife",
    "LapNumber",
    "Driver",
    "Compound",
    "lap_time_s",
]

# Use the dataset's rainfall indicator if it already exists.
# Otherwise, derive it from the Rainfall column.
if "rain_observed" not in df.columns:
    df["rain_observed"] = df["Rainfall"]

df["rain_observed"] = to_bool(df["rain_observed"])

df["Driver"] = df["Driver"].astype(str)
df["Compound"] = df["Compound"].astype(str).str.upper()

df["lap_time_s"] = pd.to_numeric(
    df["lap_time_s"], errors="coerce"
)

df["TyreLife"] = pd.to_numeric(
    df["TyreLife"], errors="coerce"
)

df["TrackTemp"] = pd.to_numeric(
    df["TrackTemp"], errors="coerce"
)

df["LapNumber"] = pd.to_numeric(
    df["LapNumber"], errors="coerce"
)

df = df[
    df["is_valid_racing_lap"]
    & df["Compound"].isin(["HARD", "MEDIUM", "SOFT"])
    & (df["lap_time_s"] > 0)
]

df = df.dropna(subset=required_columns).copy()

print(f"A1 Strict Slick records: {len(df)}")

if len(df) != 658:
    raise ValueError(
        f"Expected 658 A1 observations, found {len(df)}. "
        "Check population filtering before proceeding."
    )


# ============================================================
# 5. FIT THE BASELINE SPECIFICATION
# ============================================================

formula = (
    "lap_time_s ~ rain_observed + TrackTemp + TyreLife "
    "+ LapNumber + I(LapNumber ** 2) "
    "+ C(Driver) + C(Compound)"
)

print("\nFitting baseline regression...")

model = smf.ols(formula=formula, data=df).fit()

df["predicted_lap_time_s"] = model.predict(df)
df["residual_s"] = (
    df["lap_time_s"] - df["predicted_lap_time_s"]
)

print(f"Model observations: {int(model.nobs)}")
print(f"Model R-squared: {model.rsquared:.5f}")


# ============================================================
# 6. DEFINE RACE PHASES
# ============================================================


def classify_phase(lap):
    if lap <= 17:
        return "1. Pre-transition (laps 1-17)"
    elif lap <= 20:
        return "2. Initial wet transition (laps 18-20)"
    elif lap <= 25:
        return "3. Wet running (laps 21-25)"
    elif lap <= 27:
        return "4. Late wet / pit phase (laps 26-27)"
    else:
        return "5. Post-transition running (laps 28+)"



df["race_phase"] = df["LapNumber"].apply(classify_phase)


# ============================================================
# 7. RESIDUAL SUMMARY BY RACE PHASE
# ============================================================

phase_summary = (
    df.groupby("race_phase", observed=True)
    .agg(
        observations=("residual_s", "size"),
        unique_drivers=("Driver", "nunique"),
        first_lap=("LapNumber", "min"),
        last_lap=("LapNumber", "max"),
        mean_residual_s=("residual_s", "mean"),
        median_residual_s=("residual_s", "median"),
        residual_sd_s=("residual_s", "std"),
        mean_absolute_error_s=(
            "residual_s",
            lambda x: np.mean(np.abs(x))
        ),
        residual_rmse_s=(
            "residual_s",
            lambda x: np.sqrt(np.mean(x ** 2))
        ),
        rainfall_share=("rain_observed", "mean"),
    )
    .reset_index()
)

phase_summary.to_csv(SUMMARY_PATH, index=False)


# ============================================================
# 8. RESIDUAL SUMMARY BY INDIVIDUAL LAP
# ============================================================

lap_summary = (
    df.groupby("LapNumber")
    .agg(
        observations=("residual_s", "size"),
        unique_drivers=("Driver", "nunique"),
        mean_residual_s=("residual_s", "mean"),
        median_residual_s=("residual_s", "median"),
        residual_sd_s=("residual_s", "std"),
        mean_absolute_error_s=(
            "residual_s",
            lambda x: np.mean(np.abs(x))
        ),
        rainfall_share=("rain_observed", "mean"),
    )
    .reset_index()
)

lap_summary = lap_summary[
    lap_summary["LapNumber"].between(15, 30)
]

lap_summary.to_csv(LAP_PATH, index=False)


# ============================================================
# 9. EXPORT OBSERVATION-LEVEL RESIDUALS
# ============================================================

observation_columns = [
    "Driver",
    "LapNumber",
    "Compound",
    "TyreLife",
    "rain_observed",
    "TrackTemp",
    "lap_time_s",
    "predicted_lap_time_s",
    "residual_s",
    "race_phase",
]

df[observation_columns].to_csv(
    OBSERVATIONS_PATH,
    index=False
)


# ============================================================
# 10. PLOT RESIDUALS BY RACE PHASE
# ============================================================

phase_order = [
    "1. Pre-transition (laps 1-17)",
    "2. Initial wet transition (laps 18-20)",
    "3. Wet running (laps 21-25)",
    "4. Late wet / pit phase (laps 26-27)",
]

plot_data = [
    df.loc[df["race_phase"] == phase, "residual_s"].values
    for phase in phase_order
]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].boxplot(
    plot_data,
    tick_labels=["Pre-transition", "Initial wet", "Wet running", "Late wet / pit"],
    showfliers=True
)

axes[0].axhline(0, linestyle="--")
axes[0].set_title("Residual distribution by race phase")
axes[0].set_ylabel("Residual (seconds)")
axes[0].tick_params(axis="x", rotation=20)
axes[0].grid(axis="y", alpha=0.3)

axes[1].scatter(
    df["LapNumber"],
    df["residual_s"],
    alpha=0.45
)

axes[1].axhline(0, linestyle="--")
axes[1].set_title("Residuals across race laps")
axes[1].set_xlabel("Lap number")
axes[1].set_ylabel("Residual (seconds)")
axes[1].set_xlim(1, 30)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=200)
plt.close()


# ============================================================
# 11. PRINT RESULTS
# ============================================================

print("\nRACE-PHASE RESIDUAL SUMMARY")
print("=" * 110)

print(
    phase_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nINDIVIDUAL LAP RESIDUAL SUMMARY (LAPS 15-30)")
print("=" * 110)

print(
    lap_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\nEXPORTED FILES")
print("=" * 80)

for path in [
    SUMMARY_PATH,
    LAP_PATH,
    OBSERVATIONS_PATH,
    PLOT_PATH,
]:
    print(path)

print("\nRace-phase residual analysis completed.")

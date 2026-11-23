
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
import statsmodels.stats.api as sms

from scipy import stats


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_fia_integrated.csv"
)

OUTPUT_DIR = (
    ROOT_DIR / "data" / "processed"
    / "regression_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_PATH = (
    OUTPUT_DIR / "A1_residual_diagnostic_metrics.csv"
)

RESIDUAL_PATH = (
    OUTPUT_DIR / "A1_residual_observations.csv"
)

INFLUENCE_PATH = (
    OUTPUT_DIR / "A1_influential_observations.csv"
)

PLOT_PATH = (
    OUTPUT_DIR / "A1_residual_diagnostic_plots.png"
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def parse_bool(series):

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False
        })
    )


# ==================================================
# 3. LOAD DATA
# ==================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(INPUT_PATH)

boolean_columns = [
    "Rainfall",
    "is_deleted",
    "is_pit_transition",
    "has_valid_lap_time",
    "has_weather_match",
    "is_valid_racing_lap"
]

for col in boolean_columns:

    df[col] = parse_bool(df[col])


numeric_columns = [
    "LapNumber",
    "lap_time_s",
    "TyreLife",
    "TrackTemp"
]

for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df["Compound"] = (
    df["Compound"]
    .fillna("Unknown")
    .astype(str)
    .str.upper()
)

df["rain_observed"] = (
    df["Rainfall"].astype(float)
)


# ==================================================
# 4. PREPARE A1 STRICT SLICK POPULATION
# ==================================================

data = df[
    df["is_valid_racing_lap"].eq(True)
].copy()

data = data[
    data["Compound"].isin(
        ["HARD", "MEDIUM", "SOFT"]
    )
].copy()

required = [
    "Driver",
    "Compound",
    "lap_time_s",
    "rain_observed",
    "TrackTemp",
    "TyreLife",
    "LapNumber"
]

data = data.dropna(
    subset=required
).copy()

data = data[
    (data["lap_time_s"] > 0)
    & (data["TyreLife"] > 0)
].copy()

data["Driver"] = data["Driver"].astype(str)


print("\nDATA SUMMARY")
print("=" * 65)

print(f"Observations: {len(data)}")
print(f"Drivers: {data['Driver'].nunique()}")
print(f"Compounds: {data['Compound'].nunique()}")


# ==================================================
# 5. FIT BASELINE OLS MODEL
# ==================================================

formula = (
    "lap_time_s ~ rain_observed "
    "+ TrackTemp "
    "+ TyreLife "
    "+ LapNumber "
    "+ I(LapNumber ** 2) "
    "+ C(Driver) "
    "+ C(Compound)"
)

model = smf.ols(
    formula=formula,
    data=data
).fit()


# ==================================================
# 6. CALCULATE RESIDUALS AND INFLUENCE
# ==================================================

fitted = model.fittedvalues

residuals = model.resid

influence = model.get_influence()

studentized_residuals = (
    influence.resid_studentized_internal
)

cooks_distance = (
    influence.cooks_distance[0]
)

leverage = (
    influence.hat_matrix_diag
)

data["fitted_lap_time_s"] = fitted

data["residual_seconds"] = residuals

data["studentized_residual"] = (
    studentized_residuals
)

data["cooks_distance"] = cooks_distance

data["leverage"] = leverage


# ==================================================
# 7. RESIDUAL DISTRIBUTION TESTS
# ==================================================

print("\nRESIDUAL DISTRIBUTION")
print("=" * 65)

residual_mean = residuals.mean()

residual_std = residuals.std(ddof=1)

residual_skewness = stats.skew(residuals)

residual_kurtosis = stats.kurtosis(
    residuals,
    fisher=True
)

shapiro_stat, shapiro_p = stats.shapiro(
    residuals
)

print(f"Residual mean: {residual_mean:.6f}")
print(f"Residual std: {residual_std:.6f}")
print(f"Skewness: {residual_skewness:.6f}")
print(f"Excess kurtosis: {residual_kurtosis:.6f}")

print(f"Shapiro statistic: {shapiro_stat:.6f}")
print(f"Shapiro p-value: {shapiro_p:.6g}")


# ==================================================
# 8. BREUSCH-PAGAN TEST
# ==================================================

print("\nBREUSCH-PAGAN TEST")
print("=" * 65)

bp_lm, bp_lm_p, bp_f, bp_f_p = (
    sms.het_breuschpagan(
        residuals,
        model.model.exog
    )
)

print(f"LM statistic: {bp_lm:.6f}")
print(f"LM p-value: {bp_lm_p:.6g}")
print(f"F statistic: {bp_f:.6f}")
print(f"F p-value: {bp_f_p:.6g}")


# ==================================================
# 9. COOK'S DISTANCE
# ==================================================

parameter_count = int(model.df_model + 1)

n = len(data)

cook_threshold = 4 / n

influential = data[
    data["cooks_distance"] > cook_threshold
].copy()

print("\nINFLUENTIAL OBSERVATIONS")
print("=" * 65)

print(f"Number of parameters: {parameter_count}")
print(f"Cook's distance threshold: {cook_threshold:.6f}")

print(
    f"Observations above threshold: "
    f"{len(influential)}"
)

print(
    data.nlargest(
        10,
        "cooks_distance"
    )[
        [
            "Driver",
            "LapNumber",
            "Compound",
            "lap_time_s",
            "fitted_lap_time_s",
            "residual_seconds",
            "studentized_residual",
            "cooks_distance"
        ]
    ].to_string(index=False)
)


# ==================================================
# 10. RESIDUAL EXPORT
# ==================================================

residual_columns = [
    "Driver",
    "LapNumber",
    "Compound",
    "TyreLife",
    "TrackTemp",
    "rain_observed",
    "lap_time_s",
    "fitted_lap_time_s",
    "residual_seconds",
    "studentized_residual",
    "leverage",
    "cooks_distance"
]

data[residual_columns].to_csv(
    RESIDUAL_PATH,
    index=False
)

influential[residual_columns].to_csv(
    INFLUENCE_PATH,
    index=False
)


# ==================================================
# 11. GENERATE DIAGNOSTIC PLOTS
# ==================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(17, 10)
)

# Residual vs fitted

axes[0, 0].scatter(
    fitted,
    residuals,
    alpha=0.5
)

axes[0, 0].axhline(
    0,
    linestyle="--"
)

axes[0, 0].set_title(
    "Residuals vs Fitted"
)

axes[0, 0].set_xlabel(
    "Fitted lap time (s)"
)

axes[0, 0].set_ylabel(
    "Residual (s)"
)


# Residual histogram

axes[0, 1].hist(
    residuals,
    bins=30
)

axes[0, 1].set_title(
    "Residual Distribution"
)

axes[0, 1].set_xlabel(
    "Residual (s)"
)


# Q-Q plot

stats.probplot(
    residuals,
    dist="norm",
    plot=axes[0, 2]
)

axes[0, 2].set_title(
    "Normal Q-Q Plot"
)


# Residual vs lap number

axes[1, 0].scatter(
    data["LapNumber"],
    residuals,
    alpha=0.5
)

axes[1, 0].axhline(
    0,
    linestyle="--"
)

axes[1, 0].set_title(
    "Residuals vs Lap Number"
)

axes[1, 0].set_xlabel(
    "Lap Number"
)


# Residual vs fitted with Cook's distance highlighting

axes[1, 1].scatter(
    fitted,
    residuals,
    alpha=0.35
)

axes[1, 1].scatter(
    data.loc[
        data["cooks_distance"] > cook_threshold,
        "fitted_lap_time_s"
    ],
    data.loc[
        data["cooks_distance"] > cook_threshold,
        "residual_seconds"
    ],
    alpha=0.8,
    label="Cook's D > 4/n"
)

axes[1, 1].axhline(
    0,
    linestyle="--"
)

axes[1, 1].set_title(
    "Influential Observations"
)

axes[1, 1].legend()


# Residuals by driver

driver_residuals = [
    data.loc[
        data["Driver"] == driver,
        "residual_seconds"
    ].values
    for driver in sorted(data["Driver"].unique())
]

axes[1, 2].boxplot(
    driver_residuals,
    tick_labels=sorted(data["Driver"].unique())
)

axes[1, 2].tick_params(
    axis="x",
    rotation=90
)

axes[1, 2].set_title(
    "Residuals by Driver"
)

axes[1, 2].set_ylabel(
    "Residual (s)"
)


fig.suptitle(
    "2024 British GP – A1 Strict Slick Regression Diagnostics",
    fontsize=14
)

fig.tight_layout()

fig.savefig(
    PLOT_PATH,
    dpi=200
)

plt.close(fig)


# ==================================================
# 12. EXPORT DIAGNOSTIC METRICS
# ==================================================

metrics = {
    "sample_count": n,
    "parameter_count": parameter_count,
    "residual_mean": residual_mean,
    "residual_std": residual_std,
    "residual_skewness": residual_skewness,
    "residual_excess_kurtosis": residual_kurtosis,
    "shapiro_statistic": shapiro_stat,
    "shapiro_p_value": shapiro_p,
    "breusch_pagan_lm": bp_lm,
    "breusch_pagan_lm_p": bp_lm_p,
    "breusch_pagan_f": bp_f,
    "breusch_pagan_f_p": bp_f_p,
    "cooks_distance_threshold": cook_threshold,
    "influential_observations": len(influential),
    "maximum_cooks_distance": cooks_distance.max()
}

metrics_df = pd.DataFrame(
    [metrics]
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)


# ==================================================
# 13. FINAL REPORT
# ==================================================

print("\nDIAGNOSTIC METRICS")
print("=" * 65)

print(
    metrics_df.round(6).to_string(index=False)
)

print("\nEXPORTED FILES")
print("=" * 65)

print(METRICS_PATH)
print(RESIDUAL_PATH)
print(INFLUENCE_PATH)
print(PLOT_PATH)

print("\nResidual diagnostics completed.")


from pathlib import Path

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


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

OUTPUT_PATH = (
    OUTPUT_DIR / "race_phase_model_comparison.csv"
)


# ============================================================
# 2. LOAD AND PREPARE DATA
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

for col in ["lap_time_s", "TyreLife", "LapNumber", "TrackTemp"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Compound"] = df["Compound"].astype(str).str.upper()
df["Driver"] = df["Driver"].astype(str)

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
        "Do not continue until the population is validated."
    )


# ============================================================
# 3. ASSIGN RACE PHASES
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


df["race_phase"] = df["LapNumber"].apply(classify_phase)

phase_order = [
    "Pre-transition",
    "Initial wet transition",
    "Wet running",
    "Late wet / pit phase",
    "Post-transition running",
]

df["race_phase"] = pd.Categorical(
    df["race_phase"],
    categories=phase_order,
    ordered=True,
)


# ============================================================
# 4. DEFINE MODEL SPECIFICATIONS
# ============================================================

baseline_formula = (
    "lap_time_s ~ rain_observed + TrackTemp + TyreLife "
    "+ LapNumber + I(LapNumber ** 2) "
    "+ C(Driver) + C(Compound)"
)

phase_formula = (
    baseline_formula + " + C(race_phase)"
)


# ============================================================
# 5. FIT BOTH MODELS
# ============================================================

print("\nFitting baseline model...")

baseline_ols = smf.ols(
    formula=baseline_formula,
    data=df
).fit()

baseline_clustered = smf.ols(
    formula=baseline_formula,
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["Driver"]}
)

print("Fitting phase-adjusted model...")

phase_ols = smf.ols(
    formula=phase_formula,
    data=df
).fit()

phase_clustered = smf.ols(
    formula=phase_formula,
    data=df
).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["Driver"]}
)


# ============================================================
# 5B. DRIVER-CLUSTERED JOINT TEST OF RACE-PHASE CONTROLS
# ============================================================

print("\nDRIVER-CLUSTERED JOINT TEST OF RACE-PHASE CONTROLS")
print("=" * 80)

# Retrieve the formula design information from the
# original formula model (Patsy stores it here).
design_info = phase_clustered.model.data.orig_exog.design_info

phase_slice = design_info.term_name_slices["C(race_phase)"]

number_of_parameters = len(phase_clustered.params)
number_of_phase_parameters = (
    phase_slice.stop - phase_slice.start
)

# Construct restrictions setting all race-phase coefficients to zero.
restriction_matrix = np.zeros(
    (number_of_phase_parameters, number_of_parameters)
)

for i, column_index in enumerate(
    range(phase_slice.start, phase_slice.stop)
):
    restriction_matrix[i, column_index] = 1

# Joint test using driver-clustered covariance.
phase_wald_test = phase_clustered.wald_test(
    restriction_matrix,
    use_f=True,
    scalar=True
)

print(
    f"Number of race-phase coefficients: "
    f"{number_of_phase_parameters}"
)

print(
    f"F-statistic: {float(phase_wald_test.statistic):.6f}"
)

print(
    f"p-value: {float(phase_wald_test.pvalue):.8g}"
)

print(
    f"Numerator degrees of freedom: "
    f"{number_of_phase_parameters}"
)

print(
    f"Denominator degrees of freedom: "
    f"{getattr(phase_clustered, 'df_resid_inference', phase_clustered.df_resid):.2f}"
)


# Test all race-phase coefficients jointly using the
# driver-clustered covariance matrix.
phase_wald_test = phase_clustered.wald_test(
    restriction_matrix,
    use_f=True,
    scalar=True
)

print(
    f"Number of race-phase coefficients: "
    f"{number_of_phase_parameters}"
)

print(
    f"F-statistic: {float(phase_wald_test.statistic):.6f}"
)

print(
    f"p-value: {float(phase_wald_test.pvalue):.8g}"
)

print(
    f"Numerator degrees of freedom: "
    f"{number_of_phase_parameters}"
)

print(
    f"Denominator degrees of freedom: "
    f"{getattr(phase_clustered, 'df_resid_inference', phase_clustered.df_resid):.2f}"
)


# ============================================================
# 6. EXTRACT COMPARISON METRICS
# ============================================================


def extract_metrics(name, ols_result, clustered_result):

    # Statsmodels may encode the Boolean rainfall variable
    # as a categorical term, e.g. rain_observed[T.True].
    rainfall_terms = [
        term
        for term in clustered_result.params.index
        if term == "rain_observed"
        or term.startswith("rain_observed[")
    ]

    if len(rainfall_terms) != 1:
        raise ValueError(
            "Could not uniquely identify the rainfall coefficient. "
            f"Available terms: {list(clustered_result.params.index)}"
        )

    rainfall_term = rainfall_terms[0]

    coefficient = clustered_result.params[rainfall_term]
    standard_error = clustered_result.bse[rainfall_term]
    p_value = clustered_result.pvalues[rainfall_term]

    return {
        "model": name,
        "observations": int(ols_result.nobs),
        "drivers": df["Driver"].nunique(),
        "r_squared": ols_result.rsquared,
        "adjusted_r_squared": ols_result.rsquared_adj,
        "aic": ols_result.aic,
        "bic": ols_result.bic,
        "rainfall_coefficient_s": coefficient,
        "rainfall_clustered_se": standard_error,
        "rainfall_clustered_p_value": p_value,
        "rainfall_ci_lower": (
            coefficient - 1.96 * standard_error
        ),
        "rainfall_ci_upper": (
            coefficient + 1.96 * standard_error
        ),
    }



results = pd.DataFrame([
    extract_metrics(
        "A. Baseline",
        baseline_ols,
        baseline_clustered
    ),
    extract_metrics(
        "B. Phase-adjusted",
        phase_ols,
        phase_clustered
    ),
])

baseline_coefficient = results.loc[
    results["model"] == "A. Baseline",
    "rainfall_coefficient_s"
].iloc[0]

phase_coefficient = results.loc[
    results["model"] == "B. Phase-adjusted",
    "rainfall_coefficient_s"
].iloc[0]

results["rainfall_coefficient_change_s"] = [
    0,
    phase_coefficient - baseline_coefficient,
]

results["rainfall_coefficient_change_percent"] = [
    0,
    (
        (phase_coefficient - baseline_coefficient)
        / abs(baseline_coefficient)
    ) * 100,
]


# ============================================================
# 7. NESTED MODEL COMPARISON
# ============================================================

# Conventional partial F-test.
# This is not a driver-cluster-robust test.

f_stat, f_p_value, df_difference = (
    phase_ols.compare_f_test(baseline_ols)
)

print("\nNESTED MODEL COMPARISON")
print("=" * 80)
print(f"Partial F-statistic: {f_stat:.4f}")
print(f"Degrees of freedom difference: {df_difference}")
print(f"Conventional F-test p-value: {f_p_value:.6g}")

print(
    "\nNote: This F-test uses conventional OLS assumptions, "
    "not driver-cluster-robust inference."
)


# ============================================================
# 8. SAVE AND PRINT RESULTS
# ============================================================

results.to_csv(OUTPUT_PATH, index=False)

print("\nMODEL COMPARISON")
print("=" * 120)

print(
    results.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

print("\nEXPORTED:")
print(OUTPUT_PATH)

print("\nRace-phase model comparison completed.")


from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.formula.api as smf


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_fia_integrated.csv"
)

OUTPUT_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COEFFICIENT_PATH = (
    OUTPUT_DIR / "2024_british_gp_regression_coefficients.csv"
)

METRICS_PATH = (
    OUTPUT_DIR / "2024_british_gp_regression_metrics.csv"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "2024_british_gp_regression_predictions.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "2024_british_gp_regression_summaries.txt"
)

PLOT_PATH = (
    OUTPUT_DIR / "2024_british_gp_regression_actual_vs_predicted.html"
)


# ==================================================
# 2. HELPER FUNCTIONS
# ==================================================

def parse_bool(series):
    """Convert common CSV boolean representations to True/False."""

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
# 3. LOAD AND VALIDATE DATA
# ==================================================

print("Loading FIA-integrated dataset...")

df = pd.read_csv(INPUT_PATH)

print(f"Records loaded: {len(df)}")

required_columns = [
    "Driver",
    "LapNumber",
    "lap_time_s",
    "Compound",
    "TyreLife",
    "Rainfall",
    "TrackTemp",
    "is_deleted",
    "is_pit_transition",
    "is_accurate",
    "has_valid_lap_time",
    "has_weather_match",
    "is_valid_racing_lap"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")


# ==================================================
# 4. NORMALISE DATA TYPES
# ==================================================

boolean_columns = [
    "Rainfall",
    "is_deleted",
    "is_pit_transition",
    "is_accurate",
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

df["rain_observed"] = df["Rainfall"].astype(float)


# ==================================================
# 5. CREATE ANALYSIS POPULATIONS
# ==================================================

strict = df[
    df["is_valid_racing_lap"].eq(True)
].copy()


fia_clean = df[
    ~df["is_deleted"].fillna(False)
    & ~df["is_pit_transition"].fillna(False)
    & df["has_valid_lap_time"].fillna(False)
    & df["has_weather_match"].fillna(False)
].copy()


print("\nPOPULATION COUNTS")
print("=" * 65)

print(f"Strict: {len(strict)}")
print(f"FIA-clean: {len(fia_clean)}")


# ==================================================
# 6. PREPARE MODEL DATA
# ==================================================

def prepare_model_data(data, slick_only=False):

    model_data = data.copy()

    if slick_only:
        model_data = model_data[
            model_data["Compound"].isin(
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

    model_data = model_data.dropna(
        subset=required
    ).copy()

    model_data = model_data[
        (model_data["lap_time_s"] > 0)
        & (model_data["TyreLife"] > 0)
    ].copy()

    # Keep categorical variables consistent.
    model_data["Driver"] = (
        model_data["Driver"].astype(str)
    )

    model_data["Compound"] = (
        model_data["Compound"].astype(str)
    )

    return model_data


# ==================================================
# 7. DEFINE REGRESSION MODELS
# ==================================================

model_definitions = [
    ("A1_Strict_Slick", strict, True),
    ("A2_Strict_All", strict, False),
    ("B1_FIA_Clean_Slick", fia_clean, True),
    ("B2_FIA_Clean_All", fia_clean, False)
]


coefficient_results = []
metric_results = []
prediction_results = []
summary_text = []

fitted_primary_model = None
primary_prediction_data = None


# ==================================================
# 8. FIT REGRESSION MODELS
# ==================================================

for model_name, population, slick_only in model_definitions:

    print("\n" + "=" * 75)
    print(f"FITTING MODEL: {model_name}")
    print("=" * 75)

    model_data = prepare_model_data(
        population,
        slick_only=slick_only
    )

    if len(model_data) < 50:
        print("Insufficient records. Model skipped.")
        continue

    if model_data["rain_observed"].nunique() < 2:
        print("Both rainfall categories are not represented.")
        continue

    # --------------------------------------------------
    # Rain / compound sample counts
    # --------------------------------------------------

    print("\nRAIN / COMPOUND SAMPLE COUNTS")

    overlap = pd.crosstab(
        model_data["Compound"],
        model_data["rain_observed"]
    )

    overlap = overlap.rename(
        columns={
            0.0: "No rain observed",
            1.0: "Rain observed"
        }
    )

    print(overlap.to_string())

    # --------------------------------------------------
    # Regression formula
    # --------------------------------------------------

    formula = (
        "lap_time_s ~ rain_observed "
        "+ TrackTemp "
        "+ TyreLife "
        "+ LapNumber "
        "+ I(LapNumber ** 2) "
        "+ C(Driver) "
        "+ C(Compound)"
    )

    # --------------------------------------------------
    # Fit ordinary least squares model
    # --------------------------------------------------

    model = smf.ols(
        formula=formula,
        data=model_data
    ).fit()

    # --------------------------------------------------
    # DRIVER-CLUSTERED ROBUST STANDARD ERRORS
    # --------------------------------------------------
    #
    # Laps from the same driver may be correlated.
    # Cluster standard errors by driver to account
    # for within-driver dependence.
    #
    # Small-sample correction is enabled.
    # --------------------------------------------------

    robust = model.get_robustcov_results(
        cov_type="cluster",
        groups=model_data["Driver"].to_numpy(),
        use_correction=True,
        df_correction=True
    )

    driver_count = model_data["Driver"].nunique()

    print(
        f"\nDriver clusters used: {driver_count}"
    )

    # --------------------------------------------------
    # Extract rainfall coefficient
    # --------------------------------------------------

    param_names = list(model.params.index)

    rain_index = param_names.index(
        "rain_observed"
    )

    rain_coefficient = robust.params[rain_index]
    rain_standard_error = robust.bse[rain_index]
    rain_pvalue = robust.pvalues[rain_index]

    confidence_intervals = robust.conf_int()

    rain_ci_lower = confidence_intervals[
        rain_index, 0
    ]

    rain_ci_upper = confidence_intervals[
        rain_index, 1
    ]

    # --------------------------------------------------
    # MODEL PERFORMANCE METRICS
    # --------------------------------------------------

    predictions = model.predict(model_data)

    residuals = (
        model_data["lap_time_s"].to_numpy()
        - predictions.to_numpy()
    )

    rmse = np.sqrt(
        np.mean(residuals ** 2)
    )

    mae = np.mean(
        np.abs(residuals)
    )

    metrics = {
        "model": model_name,
        "sample_count": len(model_data),
        "driver_count": driver_count,
        "compound_count": model_data["Compound"].nunique(),
        "dry_laps": int(
            (model_data["rain_observed"] == 0).sum()
        ),
        "rain_laps": int(
            (model_data["rain_observed"] == 1).sum()
        ),
        "r_squared": model.rsquared,
        "adjusted_r_squared": model.rsquared_adj,
        "rmse_seconds_in_sample": rmse,
        "mae_seconds_in_sample": mae,
        "rain_coefficient_seconds": rain_coefficient,
        "rain_standard_error_clustered": rain_standard_error,
        "rain_ci_lower": rain_ci_lower,
        "rain_ci_upper": rain_ci_upper,
        "rain_p_value": rain_pvalue
    }

    metric_results.append(metrics)

    print("\nMODEL PERFORMANCE")

    print(
        pd.Series(metrics).to_string()
    )

    # --------------------------------------------------
    # SAVE ALL MODEL COEFFICIENTS
    # --------------------------------------------------

    for i, parameter in enumerate(param_names):

        coefficient_results.append({
            "model": model_name,
            "parameter": parameter,
            "coefficient": robust.params[i],
            "clustered_standard_error": robust.bse[i],
            "p_value": robust.pvalues[i],
            "ci_lower": confidence_intervals[i, 0],
            "ci_upper": confidence_intervals[i, 1]
        })

    # --------------------------------------------------
    # SAVE MODEL PREDICTIONS
    # --------------------------------------------------

    prediction_output = model_data[
        [
            "Driver",
            "LapNumber",
            "Compound",
            "TyreLife",
            "TrackTemp",
            "rain_observed",
            "lap_time_s"
        ]
    ].copy()

    prediction_output["model"] = model_name

    prediction_output["predicted_lap_time_s"] = (
        predictions
    )

    prediction_output["residual_seconds"] = (
        residuals
    )

    prediction_results.append(
        prediction_output
    )

    # --------------------------------------------------
    # SAVE FULL MODEL SUMMARY
    # --------------------------------------------------

    summary_text.append(
        f"\n\n{'=' * 90}\n"
        f"{model_name}\n"
        f"{'=' * 90}\n"
        f"{robust.summary().as_text()}"
    )

    # Store primary model for visualisation.
    if model_name == "A1_Strict_Slick":

        fitted_primary_model = model

        primary_prediction_data = (
            prediction_output.copy()
        )


# ==================================================
# 9. EXPORT RESULTS
# ==================================================

if not metric_results:
    raise RuntimeError(
        "No regression models were successfully fitted."
    )


coefficients_df = pd.DataFrame(
    coefficient_results
)

metrics_df = pd.DataFrame(
    metric_results
)

predictions_df = pd.concat(
    prediction_results,
    ignore_index=True
)


coefficients_df.to_csv(
    COEFFICIENT_PATH,
    index=False
)

metrics_df.to_csv(
    METRICS_PATH,
    index=False
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(summary_text)
    )


# ==================================================
# 10. ACTUAL VS PREDICTED PLOT
# ==================================================

if primary_prediction_data is not None:

    fig = px.scatter(
        primary_prediction_data,
        x="predicted_lap_time_s",
        y="lap_time_s",
        color="rain_observed",
        hover_data=[
            "Driver",
            "LapNumber",
            "Compound",
            "TyreLife"
        ],
        title=(
            "Strict Slick Model — Actual vs Predicted Lap Times"
        ),
        labels={
            "predicted_lap_time_s": (
                "Predicted lap time (seconds)"
            ),
            "lap_time_s": (
                "Actual lap time (seconds)"
            ),
            "rain_observed": (
                "Rain observed (0 = No, 1 = Yes)"
            )
        }
    )

    # Reference line: perfect prediction.
    minimum_value = min(
        primary_prediction_data[
            "predicted_lap_time_s"
        ].min(),
        primary_prediction_data[
            "lap_time_s"
        ].min()
    )

    maximum_value = max(
        primary_prediction_data[
            "predicted_lap_time_s"
        ].max(),
        primary_prediction_data[
            "lap_time_s"
        ].max()
    )

    fig.add_shape(
        type="line",
        x0=minimum_value,
        y0=minimum_value,
        x1=maximum_value,
        y1=maximum_value,
        line=dict(dash="dash")
    )

    fig.update_layout(
        template="plotly_white"
    )

    fig.write_html(
        PLOT_PATH
    )


# ==================================================
# 11. FINAL REPORT
# ==================================================

print("\n\nREGRESSION MODEL COMPARISON")
print("=" * 100)

print(
    metrics_df.round(4).to_string(index=False)
)


print("\nOUTPUT FILES")
print("=" * 65)

print(f"Coefficients: {COEFFICIENT_PATH}")
print(f"Metrics: {METRICS_PATH}")
print(f"Predictions: {PREDICTIONS_PATH}")
print(f"Full summaries: {SUMMARY_PATH}")

if primary_prediction_data is not None:
    print(
        f"Actual vs predicted plot: {PLOT_PATH}"
    )


print(
    "\nWeather regression analysis completed."
)

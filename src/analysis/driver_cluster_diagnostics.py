
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf


# ==================================================
# 1. PROJECT PATHS
# ==================================================

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_fia_integrated.csv"
)

COVERAGE_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_driver_rainfall_coverage.csv"
)

INFLUENCE_PATH = (
    ROOT_DIR / "data" / "processed"
    / "2024_british_gp_driver_cluster_influence.csv"
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


def prepare_data(population, slick_only):

    data = population.copy()

    if slick_only:

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

    return data


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

df["rain_observed"] = df["Rainfall"].astype(float)


# ==================================================
# 4. RECREATE POPULATIONS
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


populations = [
    ("A1_Strict_Slick", strict, True),
    ("A2_Strict_All", strict, False),
    ("B1_FIA_Clean_Slick", fia_clean, True),
    ("B2_FIA_Clean_All", fia_clean, False)
]


# ==================================================
# 5. BASELINE FORMULA
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


coverage_results = []
influence_results = []


# ==================================================
# 6. DRIVER CLUSTER DIAGNOSTICS
# ==================================================

for population_name, population, slick_only in populations:

    print("\n" + "=" * 80)
    print(f"POPULATION: {population_name}")
    print("=" * 80)

    data = prepare_data(
        population,
        slick_only
    )

    drivers = sorted(data["Driver"].unique())

    print(f"Observations: {len(data)}")
    print(f"Drivers: {len(drivers)}")

    # --------------------------------------------------
    # A. DRIVER RAINFALL COVERAGE
    # --------------------------------------------------

    coverage = (
        data.groupby("Driver")
        .agg(
            total_laps=("lap_time_s", "size"),
            dry_laps=(
                "rain_observed",
                lambda x: int((x == 0).sum())
            ),
            rain_laps=(
                "rain_observed",
                lambda x: int((x == 1).sum())
            )
        )
        .reset_index()
    )

    coverage["has_both_conditions"] = (
        (coverage["dry_laps"] > 0)
        & (coverage["rain_laps"] > 0)
    )

    coverage["population"] = population_name

    coverage_results.append(coverage)

    both_conditions = int(
        coverage["has_both_conditions"].sum()
    )

    print(
        f"Drivers with both conditions: "
        f"{both_conditions}/{len(drivers)}"
    )

    # --------------------------------------------------
    # B. FULL BASELINE MODEL
    # --------------------------------------------------

    baseline = smf.ols(
        formula=formula,
        data=data
    ).fit()

    baseline_coefficient = (
        baseline.params["rain_observed"]
    )

    print(
        f"Full baseline rainfall coefficient: "
        f"{baseline_coefficient:.6f}"
    )

    # --------------------------------------------------
    # C. LEAVE-ONE-DRIVER-OUT ANALYSIS
    # --------------------------------------------------

    for driver in drivers:

        reduced_data = data[
            data["Driver"] != driver
        ].copy()

        reduced_model = smf.ols(
            formula=formula,
            data=reduced_data
        ).fit()

        reduced_coefficient = (
            reduced_model.params["rain_observed"]
        )

        absolute_change = (
            reduced_coefficient
            - baseline_coefficient
        )

        percentage_change = (
            absolute_change
            / abs(baseline_coefficient)
        ) * 100

        influence_results.append({
            "population": population_name,
            "excluded_driver": driver,
            "baseline_coefficient": baseline_coefficient,
            "loo_coefficient": reduced_coefficient,
            "absolute_change": absolute_change,
            "percentage_change": percentage_change,
            "remaining_drivers": (
                reduced_data["Driver"].nunique()
            )
        })

    print(
        "Leave-one-driver-out analysis completed."
    )


# ==================================================
# 7. EXPORT RESULTS
# ==================================================

coverage_df = pd.concat(
    coverage_results,
    ignore_index=True
)

influence_df = pd.DataFrame(
    influence_results
)

coverage_df.to_csv(
    COVERAGE_PATH,
    index=False
)

influence_df.to_csv(
    INFLUENCE_PATH,
    index=False
)


# ==================================================
# 8. SUMMARY REPORT
# ==================================================

print("\nDRIVER RAINFALL COVERAGE")
print("=" * 90)

print(
    coverage_df.to_string(index=False)
)

print("\nLARGEST DRIVER INFLUENCES")
print("=" * 90)

for population_name in influence_df["population"].unique():

    subset = influence_df[
        influence_df["population"] == population_name
    ].copy()

    subset["absolute_percentage_change"] = (
        subset["percentage_change"].abs()
    )

    print(f"\n{population_name}")

    print(
        subset.sort_values(
            "absolute_percentage_change",
            ascending=False
        )[
            [
                "excluded_driver",
                "baseline_coefficient",
                "loo_coefficient",
                "percentage_change"
            ]
        ].head(5).round(5).to_string(index=False)
    )

print("\nOUTPUT FILES")
print("=" * 80)

print(COVERAGE_PATH)
print(INFLUENCE_PATH)

print("\nDriver cluster diagnostics completed.")

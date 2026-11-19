
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sqlalchemy import text

# --------------------------------------------------
# 1. Project paths and database connection
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.database.connection import get_engine

engine = get_engine()

PIT_AUDIT_PATH = ROOT_DIR / "data" / "processed" / "pit_lap_audit.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"

print("Starting validated tyre degradation analysis...")

# --------------------------------------------------
# 2. Retrieve lap and stint data
# --------------------------------------------------

query = text("""
    SELECT
        d.driver_code,
        st.stint_number,
        st.compound,
        st.start_lap,
        st.end_lap,
        st.tyre_life_start,
        l.lap_number,
        l.lap_time_ms

    FROM laps l

    JOIN stints st
        ON st.session_id = l.session_id
        AND st.driver_id = l.driver_id
        AND l.lap_number BETWEEN st.start_lap AND st.end_lap

    JOIN drivers d
        ON d.driver_id = l.driver_id

    JOIN sessions s
        ON s.session_id = l.session_id

    WHERE
        s.season_year = :year
        AND s.grand_prix = :grand_prix
        AND d.driver_code IN ('HAM', 'VER')
        AND l.lap_time_ms > 0

    ORDER BY
        d.driver_code,
        st.stint_number,
        l.lap_number
""")

with engine.connect() as connection:
    df = pd.read_sql(
        query,
        connection,
        params={
            "year": 2024,
            "grand_prix": "British Grand Prix"
        }
    )

if df.empty:
    raise ValueError("No lap records retrieved.")

print(f"Original records retrieved: {len(df)}")

# --------------------------------------------------
# 3. Calculate tyre age BEFORE pit-lap exclusion
# --------------------------------------------------

df["lap_number"] = df["lap_number"].astype(int)

df["tyre_life_start"] = pd.to_numeric(
    df["tyre_life_start"],
    errors="coerce"
)

if df["tyre_life_start"].isna().any():
    raise ValueError("Missing tyre_life_start values found.")

df["tyre_age_laps"] = (
    df["tyre_life_start"]
    + df["lap_number"]
    - df["start_lap"]
)

df["tyre_age_laps"] = df["tyre_age_laps"].astype(int)

# Validate age progression using the original lap sequence.
print("\nTYRE AGE VALIDATION")
print("=" * 75)

age_validation = []

for (driver, stint), group in df.groupby(
    ["driver_code", "stint_number"]
):
    group = group.sort_values("lap_number")

    age_steps = group["tyre_age_laps"].diff().dropna()

    valid = (
        group["tyre_age_laps"].diff().dropna().eq(
            group["lap_number"].diff().dropna()
        ).all()
    )

    age_validation.append({
        "driver": driver,
        "stint": stint,
        "first_age": group["tyre_age_laps"].iloc[0],
        "last_age": group["tyre_age_laps"].iloc[-1],
        "age_progression_valid": valid
    })

    print(
        f"{driver} | Stint {stint} | "
        f"Age {group['tyre_age_laps'].iloc[0]} "
        f"to {group['tyre_age_laps'].iloc[-1]} | "
        f"Valid: {valid}"
    )

if not all(item["age_progression_valid"] for item in age_validation):
    raise ValueError("Tyre age validation failed.")

# --------------------------------------------------
# 4. Exclude confirmed pit-in and pit-out laps
# --------------------------------------------------

if not PIT_AUDIT_PATH.exists():
    raise FileNotFoundError(
        f"Pit audit file not found: {PIT_AUDIT_PATH}"
    )

pit_audit = pd.read_csv(PIT_AUDIT_PATH)

pit_laps = set(
    zip(
        pit_audit["Driver"],
        pit_audit["LapNumber"].astype(int)
    )
)

df = df[
    ~df.apply(
        lambda row: (
            row["driver_code"],
            row["lap_number"]
        ) in pit_laps,
        axis=1
    )
].copy()

df["lap_time_s"] = df["lap_time_ms"] / 1000

print(f"\nRecords after pit-lap exclusion: {len(df)}")

# --------------------------------------------------
# 5. Calculate stint-normalised lap times
# --------------------------------------------------

group_columns = [
    "driver_code",
    "stint_number",
    "compound"
]

df["stint_median_s"] = (
    df.groupby(group_columns)["lap_time_s"]
    .transform("median")
)

df["lap_time_deviation_s"] = (
    df["lap_time_s"] - df["stint_median_s"]
)

# --------------------------------------------------
# 6. Regression and diagnostic summary
# --------------------------------------------------

summary = []

print("\nDEGRADATION TREND SUMMARY")
print("=" * 90)

for (driver, stint, compound), group in df.groupby(
    group_columns
):

    group = group.sort_values("tyre_age_laps")

    x = group["tyre_age_laps"].to_numpy(dtype=float)
    y = group["lap_time_s"].to_numpy(dtype=float)

    if len(group) < 3 or np.unique(x).size < 2:
        print(
            f"{driver} | Stint {stint}: "
            "Insufficient observations for regression."
        )
        continue

    slope, intercept = np.polyfit(x, y, 1)

    predicted = slope * x + intercept

    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    r_squared = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    summary.append({
        "driver_code": driver,
        "stint_number": stint,
        "compound": compound,
        "recorded_laps": len(group),
        "average_lap_s": y.mean(),
        "median_lap_s": np.median(y),
        "degradation_ms_per_lap": slope * 1000,
        "r_squared": r_squared
    })

    print(
        f"{driver} | Stint {stint} | {compound} | "
        f"Observations: {len(group)} | "
        f"Slope: {slope * 1000:.2f} ms/lap | "
        f"R²: {r_squared:.3f}"
    )

summary_df = pd.DataFrame(summary)

# --------------------------------------------------
# 7. Raw lap-time chart
# --------------------------------------------------

raw_fig = go.Figure()

for (driver, stint, compound), group in df.groupby(
    group_columns
):

    group = group.sort_values("tyre_age_laps")

    x = group["tyre_age_laps"].to_numpy(dtype=float)
    y = group["lap_time_s"].to_numpy(dtype=float)

    raw_fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=f"{driver} - {compound} (Stint {stint})"
        )
    )

    if len(group) >= 3 and np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)

        raw_fig.add_trace(
            go.Scatter(
                x=x,
                y=slope * x + intercept,
                mode="lines",
                line=dict(dash="dash"),
                name=f"{driver} - Stint {stint} trend"
            )
        )

raw_fig.update_layout(
    title="2024 British GP | Raw Lap Time vs Tyre Age",
    xaxis_title="Tyre Age (laps)",
    yaxis_title="Lap Time (seconds)",
    template="plotly_white",
    hovermode="x unified"
)

# --------------------------------------------------
# 8. Normalised lap-time chart
# --------------------------------------------------

normalised_fig = go.Figure()

for (driver, stint, compound), group in df.groupby(
    group_columns
):

    group = group.sort_values("tyre_age_laps")

    x = group["tyre_age_laps"].to_numpy(dtype=float)
    y = group["lap_time_deviation_s"].to_numpy(dtype=float)

    normalised_fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=f"{driver} - {compound} (Stint {stint})"
        )
    )

    if len(group) >= 3 and np.unique(x).size >= 2:
        slope, intercept = np.polyfit(x, y, 1)

        normalised_fig.add_trace(
            go.Scatter(
                x=x,
                y=slope * x + intercept,
                mode="lines",
                line=dict(dash="dash"),
                name=f"{driver} - Stint {stint} trend"
            )
        )

normalised_fig.add_hline(
    y=0,
    line_dash="dot",
    annotation_text="Stint median"
)

normalised_fig.update_layout(
    title="2024 British GP | Normalised Lap Time vs Tyre Age",
    xaxis_title="Tyre Age (laps)",
    yaxis_title="Difference from Stint Median (seconds)",
    template="plotly_white",
    hovermode="x unified"
)

# --------------------------------------------------
# 9. Export results
# --------------------------------------------------

df.to_csv(
    OUTPUT_DIR / "tyre_degradation.csv",
    index=False
)

summary_df.to_csv(
    OUTPUT_DIR / "tyre_degradation_summary.csv",
    index=False
)

raw_fig.write_html(
    OUTPUT_DIR / "tyre_degradation_raw.html"
)

normalised_fig.write_html(
    OUTPUT_DIR / "tyre_degradation_normalised.html"
)

print("\nAnalysis exports completed.")
print("CSV: tyre_degradation.csv")
print("Summary: tyre_degradation_summary.csv")
print("Chart: tyre_degradation_raw.html")
print("Chart: tyre_degradation_normalised.html")

raw_fig.show()
normalised_fig.show()


import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from sqlalchemy import text

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.database.connection import get_engine

YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "Race"

DRIVERS = ["HAM", "VER"]

engine = get_engine()


# --------------------------------------------------
# 2. LOAD LAP DATA
# --------------------------------------------------

query = text("""
    SELECT
        d.driver_code,
        l.lap_number,
        l.lap_time_ms,
        l.sector_1_ms,
        l.sector_2_ms,
        l.sector_3_ms,
        l.speed_trap_kph,
        l.is_personal_best

    FROM laps l

    JOIN drivers d
        ON l.driver_id = d.driver_id

    JOIN sessions s
        ON l.session_id = s.session_id

    WHERE s.season_year = :year
      AND s.grand_prix = :grand_prix
      AND s.session_type = :session_type
      AND d.driver_code IN ('HAM', 'VER')

    ORDER BY d.driver_code, l.lap_number
""")

with engine.connect() as connection:
    df = pd.read_sql(
        query,
        connection,
        params={
            "year": YEAR,
            "grand_prix": GRAND_PRIX,
            "session_type": SESSION_TYPE
        }
    )


# --------------------------------------------------
# 3. DATA CLEANING
# --------------------------------------------------

# Retain positive timed laps for overall lap-time analysis.
df = df[
    df["lap_time_ms"].notna()
    & (df["lap_time_ms"] > 0)
].copy()

if df.empty:
    raise ValueError("No valid timed laps found.")

# Convert milliseconds to seconds.
time_columns = [
    "lap_time_ms",
    "sector_1_ms",
    "sector_2_ms",
    "sector_3_ms"
]

for column in time_columns:
    df[column.replace("_ms", "_s")] = df[column] / 1000


# --------------------------------------------------
# 4. FULL-RACE DRIVER SUMMARY
# --------------------------------------------------

summary = (
    df.groupby("driver_code")
    .agg(
        recorded_laps=("lap_number", "count"),
        average_lap_s=("lap_time_s", "mean"),
        median_lap_s=("lap_time_s", "median"),
        fastest_lap_s=("lap_time_s", "min"),
        slowest_lap_s=("lap_time_s", "max"),
        average_sector_1_s=("sector_1_s", "mean"),
        average_sector_2_s=("sector_2_s", "mean"),
        average_sector_3_s=("sector_3_s", "mean"),
        average_speed_trap=("speed_trap_kph", "mean")
    )
    .round(3)
)

print("\nFULL RACE DRIVER COMPARISON")
print("=" * 65)
print(summary.to_string())


# --------------------------------------------------
# 5. SECTOR ADVANTAGE
# --------------------------------------------------

print("\nSECTOR TIME DIFFERENCES")
print("=" * 45)

for sector in [
    "sector_1_s",
    "sector_2_s",
    "sector_3_s"
]:

    difference = (
        summary.loc["HAM", f"average_{sector}"]
        - summary.loc["VER", f"average_{sector}"]
    )

    faster = "HAM" if difference < 0 else "VER"

    print(
        f"{sector}: {abs(difference) * 1000:.2f} ms "
        f"advantage for {faster}"
    )


# --------------------------------------------------
# 6. LAP-BY-LAP PACE GRAPH
# --------------------------------------------------

fig = px.line(
    df,
    x="lap_number",
    y="lap_time_s",
    color="driver_code",
    markers=True,
    title="2024 British GP | Full Race Lap-Time Comparison",
    labels={
        "lap_number": "Lap Number",
        "lap_time_s": "Lap Time (seconds)",
        "driver_code": "Driver"
    }
)

fig.update_layout(
    template="plotly_white",
    hovermode="x unified",
    legend_title="Driver"
)

fig.show()


# --------------------------------------------------
# 7. EXPORT ANALYSIS RESULTS
# --------------------------------------------------

output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    output_dir / "full_race_driver_summary.csv"
)

df.to_csv(
    output_dir / "full_race_lap_analysis.csv",
    index=False
)

print("\nAnalysis exports saved successfully.")
print("Full-race comparison completed.")

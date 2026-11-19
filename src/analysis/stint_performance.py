
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from sqlalchemy import text

print("Starting stint performance analysis...")

# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.database.connection import get_engine

YEAR = 2024
GRAND_PRIX = "British Grand Prix"

engine = get_engine()


# --------------------------------------------------
# 2. RETRIEVE STINT DATA
# --------------------------------------------------

query = text("""
    SELECT
        d.driver_code,
        st.stint_number,
        st.compound,
        st.start_lap,
        st.end_lap,
        l.lap_number,
        l.lap_time_ms,
        l.sector_1_ms,
        l.sector_2_ms,
        l.sector_3_ms,
        l.speed_trap_kph

    FROM stints st

    JOIN drivers d
        ON st.driver_id = d.driver_id

    JOIN sessions s
        ON st.session_id = s.session_id

    JOIN laps l
        ON l.stint_id = st.stint_id
        AND l.session_id = st.session_id
        AND l.driver_id = st.driver_id

    WHERE s.season_year = :year
      AND s.grand_prix = :grand_prix
      AND s.session_type = 'Race'
      AND d.driver_code IN ('HAM', 'VER')
      AND l.lap_time_ms > 0

    ORDER BY d.driver_code, l.lap_number
""")

with engine.connect() as connection:
    df = pd.read_sql(
        query,
        connection,
        params={
            "year": YEAR,
            "grand_prix": GRAND_PRIX
        }
    )

print("Database query completed.")
print("Records retrieved:", len(df))


# --------------------------------------------------
# 3. VALIDATE DATA
# --------------------------------------------------

if df.empty:
    raise ValueError("No stint records found in PostgreSQL.")
# Exclude confirmed pit-in and pit-out laps
pit_laps = {
    ("HAM", 27),
    ("HAM", 28),
    ("HAM", 38),
    ("HAM", 39),
    ("VER", 26),
    ("VER", 27),
    ("VER", 38),
    ("VER", 39),
}

df = df[
    ~df.apply(
        lambda row: (row["driver_code"], int(row["lap_number"]))
        in pit_laps,
        axis=1
    )
].copy()

print("\nPit-in and pit-out laps excluded.")
print(f"Remaining racing laps: {len(df)}")
df["lap_time_s"] = df["lap_time_ms"] / 1000


# --------------------------------------------------
# 4. CALCULATE STINT PERFORMANCE
# --------------------------------------------------

summary = (
    df.groupby(
        ["driver_code", "stint_number", "compound"],
        as_index=False
    )
    .agg(
        recorded_laps=("lap_number", "count"),
        average_lap_s=("lap_time_s", "mean"),
        median_lap_s=("lap_time_s", "median"),
        fastest_lap_s=("lap_time_s", "min"),
        slowest_lap_s=("lap_time_s", "max")
    )
)

print("\nSTINT PERFORMANCE SUMMARY")
print("=" * 65)
print(summary.round(3).to_string(index=False))


# --------------------------------------------------
# 5. CREATE GRAPH
# --------------------------------------------------

fig = px.bar(
    summary,
    x="compound",
    y="average_lap_s",
    color="driver_code",
    barmode="group",
    text_auto=".2f",
    title="2024 British GP | Average Lap Time by Tyre Stint",
    labels={
        "compound": "Tyre Compound",
        "average_lap_s": "Average Lap Time (seconds)",
        "driver_code": "Driver"
    }
)

fig.update_layout(
    template="plotly_white",
    legend_title="Driver"
)

fig.show()


# --------------------------------------------------
# 6. EXPORT RESULTS
# --------------------------------------------------

output_dir = ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

summary.to_csv(
    output_dir / "stint_performance_summary.csv",
    index=False
)

print("\nStint analysis completed successfully.")
print("Analysis export saved.")

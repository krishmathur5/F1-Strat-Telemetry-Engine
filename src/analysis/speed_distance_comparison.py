
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
from sqlalchemy import text

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.database.connection import get_engine

YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "Race"

DRIVERS = ["HAM", "VER"]
LAP_NUMBER = 10


# --------------------------------------------------
# 2. RETRIEVE TELEMETRY FROM POSTGRESQL
# --------------------------------------------------

engine = get_engine()

query = text("""
    SELECT
        d.driver_code,
        l.lap_number,
        t.sample_index,
        t.distance_m,
        t.speed_kph
    FROM telemetry t

    JOIN laps l
        ON t.lap_id = l.lap_id

    JOIN drivers d
        ON l.driver_id = d.driver_id

    JOIN sessions s
        ON l.session_id = s.session_id

    WHERE s.season_year = :year
      AND s.grand_prix = :grand_prix
      AND s.session_type = :session_type
      AND d.driver_code IN :drivers
      AND l.lap_number = :lap_number
      AND t.distance_m IS NOT NULL
      AND t.speed_kph IS NOT NULL

    ORDER BY
        d.driver_code,
        t.distance_m
""")

# Expand the driver list safely for SQLAlchemy
query = text(str(query).replace(
    "IN :drivers",
    "IN ('HAM', 'VER')"
))

with engine.connect() as connection:
    df = pd.read_sql(
        query,
        connection,
        params={
            "year": YEAR,
            "grand_prix": GRAND_PRIX,
            "session_type": SESSION_TYPE,
            "lap_number": LAP_NUMBER
        }
    )


# --------------------------------------------------
# 3. VALIDATE RETRIEVED DATA
# --------------------------------------------------

print("\nTELEMETRY DATA SUMMARY")
print("=" * 40)

print("Total samples:", len(df))

if df.empty:
    raise ValueError(
        "No telemetry data found. Check the session, "
        "driver codes and lap number."
    )

print("\nSamples per driver:")
print(df.groupby("driver_code").size())

print("\nDistance range:")
print(df.groupby("driver_code")["distance_m"].agg(
    ["min", "max"]
))

print("\nSpeed statistics:")
print(df.groupby("driver_code")["speed_kph"].agg(
    ["min", "max", "mean"]
))


# --------------------------------------------------
# 3B. DRIVER PERFORMANCE METRICS
# --------------------------------------------------

print("\nDRIVER PERFORMANCE COMPARISON")
print("=" * 50)

metrics = (
    df.groupby("driver_code")
    .agg(
        maximum_speed=("speed_kph", "max"),
        minimum_speed=("speed_kph", "min"),
        average_speed=("speed_kph", "mean"),
        maximum_speed_distance=("distance_m", 
                                 lambda x: x.loc[
                                     df.loc[x.index, "speed_kph"].idxmax()
                                 ])
    )
    .round(2)
)

print(metrics)

# Compare maximum speed
if all(driver in metrics.index for driver in DRIVERS):

    ham_speed = metrics.loc["HAM", "maximum_speed"]
    ver_speed = metrics.loc["VER", "maximum_speed"]

    print("\nMaximum speed difference:")
    print(f"HAM - VER: {ham_speed - ver_speed:.2f} km/h")



# --------------------------------------------------
# 4. CREATE SPEED VS DISTANCE GRAPH
# --------------------------------------------------

fig = px.line(
    df,
    x="distance_m",
    y="speed_kph",
    color="driver_code",
    title=(
        f"Speed vs Distance: HAM vs VER "
        f"| 2024 British GP | Lap {LAP_NUMBER}"
    ),
    labels={
        "distance_m": "Distance along lap (m)",
        "speed_kph": "Speed (km/h)",
        "driver_code": "Driver"
    }
)

fig.update_layout(
    template="plotly_white",
    xaxis_title="Distance (m)",
    yaxis_title="Speed (km/h)",
    hovermode="x unified",
    legend_title="Driver",
)

fig.show()

print("\nSpeed comparison graph generated successfully.")

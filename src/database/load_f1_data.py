
import sys
from pathlib import Path

import fastf1
import pandas as pd
from sqlalchemy import text

# --------------------------------------------------
# 1. PROJECT PATHS AND DATABASE CONNECTION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from src.database.connection import get_engine

CSV_FILE = ROOT / "data" / "processed" / "2024_british_gp_laps_processed.csv"

YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "R"

# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------


def clean_value(value):
    """Convert pandas/numpy values into PostgreSQL-compatible values."""
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def seconds_to_ms(value):
    """Convert seconds into milliseconds."""
    value = clean_value(value)

    if value is None:
        return None

    return float(value) * 1000


def clean_integer(value):
    """Convert a numeric value into an integer or None."""
    value = clean_value(value)

    if value is None:
        return None

    return int(value)

# --------------------------------------------------
# 3. LOAD SOURCE DATA
# --------------------------------------------------

print("Reading processed lap data...")

df = pd.read_csv(CSV_FILE)

print(f"Loaded {len(df)} lap records.")

print("Retrieving FastF1 session metadata...")

session = fastf1.get_session(
    YEAR,
    GRAND_PRIX,
    SESSION_TYPE
)

session.load(
    laps=False,
    telemetry=False,
    weather=False,
    messages=False
)

event = session.event
driver_results = session.results

# --------------------------------------------------
# 4. PREPARE STINT DATA
# --------------------------------------------------

stint_data = (
    df.groupby(["driver_code", "stint_number"])
    .agg(
        start_lap=("lap_number", "min"),
        end_lap=("lap_number", "max"),
        compound=("tyre_compound", "first"),
        tyre_life_start=("tyre_life", "min")
    )
    .reset_index()
)

print(f"Identified {len(stint_data)} unique stints.")

# --------------------------------------------------
# 5. DATABASE LOADING
# --------------------------------------------------

engine = get_engine()

with engine.begin() as conn:

    print("\n--- LOADING CIRCUIT ---")

    circuit_name = f"{event['Location']} Circuit"

    circuit_id = conn.execute(
        text("""
            SELECT circuit_id
            FROM circuits
            WHERE circuit_name = :name
              AND location = :location
        """),
        {
            "name": circuit_name,
            "location": event["Location"]
        }
    ).scalar_one_or_none()

    if circuit_id is None:
        circuit_id = conn.execute(
            text("""
                INSERT INTO circuits (
                    circuit_name,
                    location,
                    country
                )
                VALUES (
                    :name,
                    :location,
                    :country
                )
                RETURNING circuit_id
            """),
            {
                "name": circuit_name,
                "location": event["Location"],
                "country": event["Country"]
            }
        ).scalar_one()

    print("Circuit ID:", circuit_id)

    # --------------------------------------------------
    # 6. LOAD SESSION
    # --------------------------------------------------

    print("\n--- LOADING SESSION ---")

    session_id = conn.execute(
        text("""
            SELECT session_id
            FROM sessions
            WHERE season_year = :year
              AND grand_prix = :grand_prix
              AND session_type = :session_type
        """),
        {
            "year": YEAR,
            "grand_prix": GRAND_PRIX,
            "session_type": "Race"
        }
    ).scalar_one_or_none()

    if session_id is None:
        session_id = conn.execute(
            text("""
                INSERT INTO sessions (
                    circuit_id,
                    season_year,
                    grand_prix,
                    session_type,
                    session_date
                )
                VALUES (
                    :circuit_id,
                    :year,
                    :grand_prix,
                    :session_type,
                    :session_date
                )
                RETURNING session_id
            """),
            {
                "circuit_id": circuit_id,
                "year": YEAR,
                "grand_prix": GRAND_PRIX,
                "session_type": "Race",
                "session_date": event["EventDate"].date()
            }
        ).scalar_one()

    print("Session ID:", session_id)

    # --------------------------------------------------
    # 7. LOAD DRIVERS
    # --------------------------------------------------

    print("\n--- LOADING DRIVERS ---")

    driver_ids = {}

    for _, driver in driver_results.iterrows():

        code = driver["Abbreviation"]
        full_name = driver["FullName"]

        driver_id = conn.execute(
            text("""
                SELECT driver_id
                FROM drivers
                WHERE driver_code = :code
            """),
            {"code": code}
        ).scalar_one_or_none()

        if driver_id is None:
            driver_id = conn.execute(
                text("""
                    INSERT INTO drivers (
                        driver_code,
                        full_name
                    )
                    VALUES (
                        :code,
                        :full_name
                    )
                    RETURNING driver_id
                """),
                {
                    "code": code,
                    "full_name": full_name
                }
            ).scalar_one()

        driver_ids[code] = driver_id

    print(f"Drivers available: {len(driver_ids)}")

    # --------------------------------------------------
    # 8. LOAD STINTS
    # --------------------------------------------------

    print("\n--- LOADING STINTS ---")

    stint_ids = {}
    new_stints = 0

    for _, stint in stint_data.iterrows():

        code = stint["driver_code"]
        number = int(stint["stint_number"])
        driver_id = driver_ids[code]

        stint_id = conn.execute(
            text("""
                SELECT stint_id
                FROM stints
                WHERE session_id = :session_id
                  AND driver_id = :driver_id
                  AND stint_number = :number
            """),
            {
                "session_id": session_id,
                "driver_id": driver_id,
                "number": number
            }
        ).scalar_one_or_none()

        if stint_id is None:

            stint_id = conn.execute(
                text("""
                    INSERT INTO stints (
                        session_id,
                        driver_id,
                        stint_number,
                        compound,
                        start_lap,
                        end_lap,
                        tyre_life_start
                    )
                    VALUES (
                        :session_id,
                        :driver_id,
                        :number,
                        :compound,
                        :start_lap,
                        :end_lap,
                        :tyre_life_start
                    )
                    RETURNING stint_id
                """),
                {
                    "session_id": session_id,
                    "driver_id": driver_id,
                    "number": number,
                    "compound": clean_value(stint["compound"]),
                    "start_lap": int(stint["start_lap"]),
                    "end_lap": int(stint["end_lap"]),
                    "tyre_life_start": clean_value(
                        stint["tyre_life_start"]
                    )
                }
            ).scalar_one()

            new_stints += 1

        stint_ids[(code, number)] = stint_id

    print("New stints inserted:", new_stints)
    print("Total stints available:", len(stint_ids))

    # --------------------------------------------------
    # 9. LOAD LAPS
    # --------------------------------------------------

    print("\n--- LOADING LAPS ---")

    new_laps = 0
    existing_laps = 0

    for _, lap in df.iterrows():

        code = lap["driver_code"]
        driver_id = driver_ids[code]

        lap_number = int(lap["lap_number"])

        stint_number = int(lap["stint_number"])
        stint_id = stint_ids[(code, stint_number)]

        result = conn.execute(
            text("""
                INSERT INTO laps (
                    session_id,
                    driver_id,
                    stint_id,
                    lap_number,
                    lap_time_ms,
                    sector_1_ms,
                    sector_2_ms,
                    sector_3_ms,
                    speed_trap_kph,
                    position,
                    is_personal_best
                )
                VALUES (
                    :session_id,
                    :driver_id,
                    :stint_id,
                    :lap_number,
                    :lap_time_ms,
                    :sector_1_ms,
                    :sector_2_ms,
                    :sector_3_ms,
                    :speed_trap_kph,
                    :position,
                    :is_personal_best
                )
                ON CONFLICT (
                    session_id,
                    driver_id,
                    lap_number
                )
                DO NOTHING
                RETURNING lap_id
            """),
            {
                "session_id": session_id,
                "driver_id": driver_id,
                "stint_id": stint_id,
                "lap_number": lap_number,
                "lap_time_ms": seconds_to_ms(lap["lap_time"]),
                "sector_1_ms": seconds_to_ms(lap["sector1_time"]),
                "sector_2_ms": seconds_to_ms(lap["sector2_time"]),
                "sector_3_ms": seconds_to_ms(lap["sector3_time"]),
                "speed_trap_kph": clean_value(lap["speed_st"]),
                "position": clean_integer(lap["position"]),python -u src/database/load_f1_data.py
                "is_personal_best": clean_value(
                    lap["is_personal_best"]
                )
            }
        ).scalar_one_or_none()

        if result is None:
            existing_laps += 1
        else:
            new_laps += 1

    print("New laps inserted:", new_laps)
    print("Existing laps skipped:", existing_laps)

# --------------------------------------------------
# 10. VERIFY DATABASE
# --------------------------------------------------

print("\n--- DATABASE VERIFICATION ---")

with engine.connect() as conn:

    for table in [
        "circuits",
        "sessions",
        "drivers",
        "stints",
        "laps",
        "telemetry"
    ]:

        count = conn.execute(
            text(f"SELECT COUNT(*) FROM {table}")
        ).scalar_one()

        print(f"{table}: {count}")

print("\nDATABASE LOADING COMPLETED SUCCESSFULLY.")

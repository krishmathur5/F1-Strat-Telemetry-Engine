
import sys
from pathlib import Path

import fastf1
import pandas as pd
from sqlalchemy import text

# --------------------------------------------------
# 1. PROJECT CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))

from src.database.connection import get_engine

YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "R"

BATCH_SIZE = 500

# Project-local FastF1 cache
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def clean_value(value):
    """Convert missing and NumPy values to Python values."""

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def prepare_telemetry(telemetry, lap_id):
    """Convert FastF1 telemetry into database records."""

    required_columns = [
        "Distance",
        "Speed",
        "Throttle",
        "Brake",
        "nGear",
        "RPM",
        "DRS"
    ]

    telemetry = telemetry[required_columns].copy()

    telemetry = telemetry.reset_index(drop=True)

    records = []

    for sample_index, row in telemetry.iterrows():

        records.append({
            "lap_id": int(lap_id),
            "sample_index": int(sample_index),
            "distance_m": clean_value(row["Distance"]),
            "speed_kph": clean_value(row["Speed"]),
            "throttle_pct": clean_value(row["Throttle"]),
            "brake": clean_value(row["Brake"]),
            "gear": (
                int(row["nGear"])
                if pd.notna(row["nGear"])
                else None
            ),
            "rpm": (
                int(row["RPM"])
                if pd.notna(row["RPM"])
                else None
            ),
            "drs": (
                int(row["DRS"])
                if pd.notna(row["DRS"])
                else None
            )
        })

    return records


# --------------------------------------------------
# 3. LOAD FASTF1 SESSION
# --------------------------------------------------

print("\nLoading British Grand Prix session...")

session = fastf1.get_session(
    YEAR,
    GRAND_PRIX,
    SESSION_TYPE
)

session.load(
    laps=True,
    telemetry=True,
    weather=False,
    messages=False
)

print("FastF1 session loaded.")


# --------------------------------------------------
# 4. BUILD FASTF1 LAP LOOKUP
# --------------------------------------------------

print("\nBuilding FastF1 lap lookup...")

fastf1_laps = {}

for _, lap in session.laps.iterrows():

    if pd.isna(lap["LapNumber"]):
        continue

    driver_code = lap["Driver"]
    lap_number = int(lap["LapNumber"])

    fastf1_laps[(driver_code, lap_number)] = lap

print("FastF1 laps available:", len(fastf1_laps))


# --------------------------------------------------
# 5. RETRIEVE DATABASE LAP RECORDS
# --------------------------------------------------

engine = get_engine()

with engine.connect() as conn:

    database_laps = conn.execute(
        text("""
            SELECT
                l.lap_id,
                d.driver_code,
                l.lap_number
            FROM laps l
            JOIN drivers d
                ON l.driver_id = d.driver_id
            JOIN sessions s
                ON l.session_id = s.session_id
            WHERE s.season_year = :year
              AND s.grand_prix = :grand_prix
              AND s.session_type = 'Race'
            ORDER BY d.driver_code, l.lap_number
        """),
        {
            "year": YEAR,
            "grand_prix": GRAND_PRIX
        }
    ).mappings().all()

print("Database laps available:", len(database_laps))


# --------------------------------------------------
# 6. PREPARE SQL INSERT
# --------------------------------------------------

insert_sql = text("""
    INSERT INTO telemetry (
        lap_id,
        sample_index,
        distance_m,
        speed_kph,
        throttle_pct,
        brake,
        gear,
        rpm,
        drs
    )
    VALUES (
        :lap_id,
        :sample_index,
        :distance_m,
        :speed_kph,
        :throttle_pct,
        :brake,
        :gear,
        :rpm,
        :drs
    )
    ON CONFLICT (lap_id, sample_index)
    DO NOTHING
""")


# --------------------------------------------------
# 7. EXTRACT AND INSERT TELEMETRY
# --------------------------------------------------

print("\nStarting telemetry extraction...")

successful_laps = 0
skipped_laps = 0
failed_laps = 0
total_samples_inserted = 0

failures = []

for index, db_lap in enumerate(database_laps, start=1):

    lap_id = db_lap["lap_id"]
    driver_code = db_lap["driver_code"]
    lap_number = int(db_lap["lap_number"])

    lap_key = (driver_code, lap_number)

    print(
        f"\n[{index}/{len(database_laps)}] "
        f"{driver_code} - Lap {lap_number}",
        flush=True
    )

    try:

        # Find corresponding FastF1 lap
        fastf1_lap = fastf1_laps.get(lap_key)

        if fastf1_lap is None:
            raise ValueError("Matching FastF1 lap not found.")

        # Check whether this lap was already extracted
        with engine.connect() as conn:

            existing_samples = conn.execute(
                text("""
                    SELECT COUNT(*)
                    FROM telemetry
                    WHERE lap_id = :lap_id
                """),
                {"lap_id": lap_id}
            ).scalar_one()

        if existing_samples > 0:

            print(
                f"Already loaded: {existing_samples} samples. Skipping."
            )

            skipped_laps += 1
            continue

        # Extract telemetry
        telemetry = fastf1_lap.get_telemetry()

        if telemetry.empty:
            raise ValueError("No telemetry samples available.")

        records = prepare_telemetry(
            telemetry,
            lap_id
        )

        # Insert in batches within one transaction per lap
        inserted_for_lap = 0

        with engine.begin() as conn:

            for start in range(0, len(records), BATCH_SIZE):

                batch = records[
                    start:start + BATCH_SIZE
                ]

                result = conn.execute(
                    insert_sql,
                    batch
                )

                inserted_for_lap += result.rowcount

        successful_laps += 1
        total_samples_inserted += inserted_for_lap

        print(
            f"Inserted {inserted_for_lap} telemetry samples."
        )

    except Exception as error:

        failed_laps += 1

        failures.append({
            "driver": driver_code,
            "lap": lap_number,
            "error": str(error)
        })

        print("EXTRACTION FAILED:", error)


# --------------------------------------------------
# 8. FINAL SUMMARY
# --------------------------------------------------

print("\n" + "=" * 50)
print("TELEMETRY EXTRACTION SUMMARY")
print("=" * 50)

print("Database laps:", len(database_laps))
print("Successfully processed:", successful_laps)
print("Already loaded/skipped:", skipped_laps)
print("Failed laps:", failed_laps)
print("New telemetry samples:", total_samples_inserted)

if failures:

    print("\n--- FAILED EXTRACTIONS ---")

    for failure in failures:
        print(failure)

# Final database count
with engine.connect() as conn:

    total_database_samples = conn.execute(
        text("SELECT COUNT(*) FROM telemetry")
    ).scalar_one()

print("\nTotal telemetry records in PostgreSQL:")
print(total_database_samples)

print("\nTelemetry extraction process completed.")


from pathlib import Path
import fastf1

ROOT_DIR = Path(__file__).resolve().parents[2]

CACHE_DIR = ROOT_DIR / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

print("Loading British GP race timing and race-control messages...")

session = fastf1.get_session(
    2024,
    "British Grand Prix",
    "R"
)

session.load(
    laps=True,
    telemetry=False,
    weather=False,
    messages=True
)

laps = session.laps.copy()

print("\nAVAILABLE DELETION FIELDS")
print("=" * 65)

print(laps[["Deleted", "DeletedReason"]].dtypes)

print("\nDELETED LAP COUNT")
print("=" * 65)

print(laps["Deleted"].value_counts(dropna=False))

deleted_laps = laps[
    laps["Deleted"].eq(True)
].copy()

print("\nDELETED LAP DETAILS")
print("=" * 100)

if deleted_laps.empty:
    print("No deleted laps identified by FastF1.")
else:
    print(
        deleted_laps[
            [
                "Driver",
                "LapNumber",
                "LapTime",
                "Deleted",
                "DeletedReason",
                "TrackStatus"
            ]
        ].to_string(index=False)
    )

output_path = (
    ROOT_DIR
    / "data" / "processed"
    / "2024_british_gp_deleted_laps.csv"
)

deleted_laps.to_csv(output_path, index=False)

print("\nExtraction completed.")
print(f"Deleted-lap export: {output_path}")

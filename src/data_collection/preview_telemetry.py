
import fastf1
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

fastf1.Cache.enable_cache(str(ROOT / "data" / "cache"))
YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "R"

print("Loading race session...")

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

# Select Lewis Hamilton's first recorded lap
lap = session.laps[
    (session.laps["Driver"] == "HAM") &
    (session.laps["LapNumber"] == 1)
].iloc[0]

print("\nSelected lap:")
print("Driver:", lap["Driver"])
print("Lap number:", lap["LapNumber"])

print("\nExtracting telemetry...")

telemetry = lap.get_telemetry()

required_columns = [
    "Distance",
    "Speed",
    "Throttle",
    "Brake",
    "nGear",
    "RPM",
    "DRS"
]

print("\nAvailable telemetry columns:")
print(telemetry.columns.tolist())

print("\nTelemetry preview:")
print(telemetry[required_columns].head(10).to_string(index=False))

print("\nTotal telemetry samples:", len(telemetry))

print("\nTelemetry preview completed.")


import fastf1

print("SCRIPT STARTED", flush=True)

YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "R"

print("Connecting to FastF1...", flush=True)

session = fastf1.get_session(
    YEAR,
    GRAND_PRIX,
    SESSION_TYPE
)

print("Loading session metadata...", flush=True)

session.load(
    laps=False,
    telemetry=False,
    weather=False,
    messages=False
)

print("\n--- EVENT INFORMATION ---")
print(session.event)

print("\n--- DRIVER INFORMATION ---")

driver_info = session.results[
    ["DriverNumber", "Abbreviation", "FullName", "TeamName"]
]

print(driver_info.to_string(index=False))

print("\nMETADATA INSPECTION COMPLETED")

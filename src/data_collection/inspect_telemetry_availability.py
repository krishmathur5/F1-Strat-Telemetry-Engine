
import fastf1

YEAR = 2024
GRAND_PRIX = "British Grand Prix"
SESSION_TYPE = "R"

print("Loading British GP race data...")

session = fastf1.get_session(
    YEAR,
    GRAND_PRIX,
    SESSION_TYPE
)

session.load(
    laps=True,
    telemetry=False,
    weather=False,
    messages=False
)

laps = session.laps

print("\n--- DRIVER LAP AVAILABILITY ---")

driver_summary = (
    laps.groupby("Driver")
    .agg(
        recorded_laps=("LapNumber", "count"),
        first_lap=("LapNumber", "min"),
        last_lap=("LapNumber", "max")
    )
    .reset_index()
)

print(driver_summary.to_string(index=False))

print("\n--- GASLY INVESTIGATION ---")

gasly_laps = laps[
    laps["Driver"] == "GAS"
]

print("Gasly recorded laps:", len(gasly_laps))

if len(gasly_laps) > 0:
    print(gasly_laps.to_string(index=False))
else:
    print("No recorded laps found for GAS.")

print("\n--- SESSION SUMMARY ---")
print("Drivers with lap data:", laps["Driver"].nunique())
print("Total recorded laps:", len(laps))

print("\nInspection completed.")

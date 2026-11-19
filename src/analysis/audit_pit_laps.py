from pathlib import Path
import pandas as pd

# Locate the original FastF1 dataset
ROOT_DIR = Path(__file__).resolve().parents[2]

CSV_PATH = ROOT_DIR / "data" / "raw" / "2024_british_gp_laps.csv"

print("Loading original FastF1 lap data...")

df = pd.read_csv(
    CSV_PATH,
    na_values=["NaT", "nat", ""]
)

# Check that the required columns exist
required_columns = [
    "Driver",
    "LapNumber",
    "PitInTime",
    "PitOutTime",
    "Stint",
    "Compound"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# Identify laps with pit-in or pit-out timestamps
pit_laps = df[
    df["PitInTime"].notna() |
    df["PitOutTime"].notna()
].copy()

# Focus on Hamilton and Verstappen
pit_laps = pit_laps[
    pit_laps["Driver"].isin(["HAM", "VER"])
]

pit_laps = pit_laps.sort_values(
    ["Driver", "LapNumber"]
)

print("\nPIT-IN / PIT-OUT LAP AUDIT")
print("=" * 75)

if pit_laps.empty:
    print("No pit timestamps found for HAM or VER.")
else:
    print(
        pit_laps[
            [
                "Driver",
                "LapNumber",
                "Stint",
                "Compound",
                "PitInTime",
                "PitOutTime"
            ]
        ].to_string(index=False)
    )

# Export the audit
output_path = (
    ROOT_DIR / "data" / "processed" / "pit_lap_audit.csv"
)

pit_laps.to_csv(output_path, index=False)

print("\nAudit completed.")
print(f"Export saved to: {output_path}")
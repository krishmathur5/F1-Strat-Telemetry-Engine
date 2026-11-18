
import pandas as pd

FILE_PATH = "data/processed/2024_british_gp_laps_processed.csv"

df = pd.read_csv(FILE_PATH)

print("\n--- DATASET SUMMARY ---")
print("Total laps:", len(df))
print("Drivers:", df["driver_code"].nunique())
print("Unique stints:", df.groupby(
    ["driver_code", "stint_number"]
).ngroups)

print("\n--- STINT SUMMARY ---")

stints = (
    df.groupby(
        ["driver_code", "stint_number"],
        dropna=False
    )
    .agg(
        start_lap=("lap_number", "min"),
        end_lap=("lap_number", "max"),
        compound=("tyre_compound", "first"),
        tyre_life_start=("tyre_life", "min"),
        laps_in_stint=("lap_number", "count")
    )
    .reset_index()
)

print(stints.to_string(index=False))

print("\n--- COMPOUNDS ---")
print(df["tyre_compound"].value_counts(dropna=False))

print("\nStint inspection completed.")

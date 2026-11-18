
import fastf1
import pandas as pd
from pathlib import Path

Path("data/cache").mkdir(parents=True, exist_ok=True)


fastf1.Cache.enable_cache("data/cache")


session = fastf1.get_session(2024, "Abu Dhabi", "R")


print("Loading Formula 1 race data...")
session.load()


laps = session.laps


print("\nSession loaded successfully!")
print("Total laps recorded:", len(laps))

print("\nAvailable columns:")
print(laps.columns.tolist())


print("\nFirst five laps:")
print(laps.head())


laps.to_csv("data/raw/abu_dhabi_2024_laps.csv", index=False)

print("\nData exported successfully!")

import glob
import os

import pandas as pd

BRANDS = {"merc": "mercedes", "vw": "volkswagen", "hyundi": "hyundai"}
SKIP = ("unclean", "cclass", "focus")  # duplicate subset files in the dataset

files = glob.glob("data/raw/*.csv")
if not files:
    raise SystemExit("No CSV files in data/raw/. Download the dataset first (see README).")

frames = []
for path in files:
    name = os.path.splitext(os.path.basename(path))[0]
    if any(s in name.lower() for s in SKIP):
        continue
    d = pd.read_csv(path)
    d.columns = [c.strip().replace("tax(£)", "tax") for c in d.columns]
    d["brand"] = BRANDS.get(name, name)
    frames.append(d)

df = pd.concat(frames, ignore_index=True)
print("Raw rows:", len(df))
print("Columns:", list(df.columns))

for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].str.strip()

df = df.drop_duplicates()
df = df.dropna()
df = df[(df["price"] > 0) & (df["year"].between(1995, 2020))]
df = df[(df["engineSize"] > 0) & (df["mileage"] >= 0) & (df["mpg"] > 0)]
low, high = df["price"].quantile([0.01, 0.99])
df = df[df["price"].between(low, high)]

print("Clean rows:", len(df))
os.makedirs("data/processed", exist_ok=True)
df.to_csv("data/processed/cars_clean.csv", index=False)
print("Saved data/processed/cars_clean.csv")

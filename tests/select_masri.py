import pandas as pd

INPUT = "data/masri30/metadata.csv"
OUTPUT = "data/masri30/selected_metadata.csv"

TARGET = 500

df = pd.read_csv(INPUT)

# Quality filters
df = df[
    (df["split"] == "train") &
    (df["confidence"] >= 0.95) &
    (df["promo"] == False) &
    (df["duration"] >= 3)
]

print("=" * 50)
print("EGYPTIAN ARABIC DATASET")
print("=" * 50)

print(f"Available clips: {len(df)}")
print(f"Available hours: {df['duration'].sum() / 3600:.2f}")
print(f"Available videos: {df['video_id'].nunique()}")

# Shuffle
df = df.sample(frac=1, random_state=42)

# Select exactly 500 clips
selected = df.head(TARGET)

selected.to_csv(
    OUTPUT,
    index=False
)

print()
print(f"Samples: {len(selected)}")
print(f"Duration: {selected['duration'].sum() / 60:.2f} minutes")
print(f"Hours: {selected['duration'].sum() / 3600:.2f}")
print(f"Videos: {selected['video_id'].nunique()}")
print(f"Output: {OUTPUT}")
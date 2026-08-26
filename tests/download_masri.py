import os
import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "ehabnegm/100-hour-Egyption-dataset-single-speaker"
REPO_TYPE = "dataset"

CSV_PATH = "data/masri30/selected_metadata.csv"
OUTPUT_DIR = "data/masri30/wavs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

print("=" * 60)
print("EGYPTIAN ARABIC WAV DOWNLOADER")
print("=" * 60)
print(f"Files: {len(df)}")
print(f"Output: {OUTPUT_DIR}")
print()

for i, row in df.iterrows():

    # Path inside Hugging Face dataset
    file_name = row["file_name"]

    # Extract just the WAV filename
    output_name = os.path.basename(file_name)

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    print(f"[{i + 1}/{len(df)}] {file_name}")

    if os.path.exists(output_path):
        print("  Already downloaded.")
        continue

    try:

        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            filename=file_name,
        )

        # Copy actual WAV file
        with open(downloaded_path, "rb") as src:
            with open(output_path, "wb") as dst:
                dst.write(src.read())

        print("  ✓ WAV downloaded")

    except Exception as e:

        print(f"  ✗ ERROR: {e}")

print()
print("=" * 60)
print("DONE")
print("=" * 60)

wav_files = [
    f for f in os.listdir(OUTPUT_DIR)
    if f.lower().endswith(".wav")
]

print(f"WAV files in directory: {len(wav_files)}")
print(f"Location: {OUTPUT_DIR}")
import os
import pandas as pd

def clean_csv(csv_path, audio_dir, output_path):
    """
    Removes entries from CSV whose audio files are missing.
    """
    df = pd.read_csv(csv_path)
    existing = {os.path.splitext(f)[0] for f in os.listdir(audio_dir)}
    df = df[df["filename"].isin(existing)]
    df.to_csv(output_path, index=False)
    print(f"Cleaned CSV saved to {output_path} (kept {len(df)} files)")

if __name__ == "__main__":
    clean_csv("data/train.csv", "audio/train", "data/train_clean.csv")

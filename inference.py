import os
import torch
import pandas as pd
import numpy as np
import librosa
from utils import load_audio
import torch.nn as nn

# -------------------- Model --------------------
class SimpleAudioModel(nn.Module):
    def __init__(self, input_dim=59):
        super(SimpleAudioModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- Config --------------------
audio_dir = "audio/test"
test_csv = "data/test.csv"
output_csv = "data/submission.csv"
model_path = "models/final_model.pth"

# -------------------- Load Model --------------------
model = SimpleAudioModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# -------------------- Predict --------------------
df = pd.read_csv(test_csv)
predictions = []

for i, row in df.iterrows():
    fname = row["filename"]
    file_path = os.path.join(audio_dir, f"{fname}.wav")

    audio = load_audio(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=59)  # ===== FIX: 59
    mfcc = np.mean(mfcc.T, axis=0)
    x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(x).item()

    predictions.append(pred)

df["predicted_score"] = predictions
df.to_csv(output_csv, index=False)
print(f"Predictions saved to {output_csv}")

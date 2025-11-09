import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils import load_audio
import numpy as np
import librosa
from tqdm import tqdm

# -------------------- Load Cleaned Data --------------------
csv_path = "data/train.csv"
if os.path.exists("data/train_clean.csv"):
    csv_path = "data/train_clean.csv"

df = pd.read_csv(csv_path)
audio_dir = "audio/train"

# -------------------- Audio Dataset --------------------
class AudioDataset(Dataset):
    def __init__(self, df, audio_dir, sr=16000):
        self.df = df
        self.audio_dir = audio_dir
        self.sr = sr

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = os.path.join(self.audio_dir, f"{row['filename']}.wav")
        audio = load_audio(filepath, sr=self.sr)

        # ===== FIX: Match checkpoint dimension =====
        mfcc = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=59)
        mfcc = np.mean(mfcc.T, axis=0)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return torch.tensor(mfcc, dtype=torch.float32), label

# -------------------- Model --------------------
class SimpleAudioModel(nn.Module):
    def __init__(self, input_dim=59):  # ===== FIX: input_dim=59
        super(SimpleAudioModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # ===== FIX: add batchnorm to match checkpoint
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------- Training --------------------
def train_model():
    X_train, X_val = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = AudioDataset(X_train, audio_dir)
    val_dataset = AudioDataset(X_val, audio_dir)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = SimpleAudioModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/5"):
            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/final_model.pth")
    print("Model saved to models/final_model.pth")


if __name__ == "__main__":
    train_model()

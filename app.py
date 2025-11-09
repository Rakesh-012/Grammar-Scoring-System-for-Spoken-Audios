import os
import torch
import numpy as np
import librosa
from flask import Flask, render_template, request, redirect, url_for
from train_model import SimpleAudioModel
from utils import load_audio

# -------------------- Flask Setup --------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
MODEL_PATH = "models/final_model.pth"

# -------------------- Load Model --------------------
model = SimpleAudioModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# -------------------- Helper: Predict Score --------------------
def predict_score(filepath):
    try:
        audio = load_audio(filepath)
        mfcc = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            score = model(x).item()
        return round(score, 2)
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "audiofile" not in request.files:
            return redirect(request.url)
        file = request.files["audiofile"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            score = predict_score(filepath)
            return render_template("index.html", score=score, filename=file.filename)

    return render_template("index.html", score=None)

# -------------------- Run --------------------
if __name__ == "__main__":
    app.run(debug=True)

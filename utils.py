import librosa
import numpy as np

def load_audio(filepath, sr=16000):
    """
    Loads an audio file safely. 
    If missing or corrupted, returns 1 sec of silence instead of crashing.
    """
    try:
        audio, _ = librosa.load(filepath, sr=sr)
        return audio
    except Exception as e:
        print(f"Skipping file {filepath}: {e}")
        return np.zeros(sr)

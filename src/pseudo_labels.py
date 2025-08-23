import numpy as np
import librosa

def audio_to_pseudo_labels(audio_path, sr=16000, hop_length=160, fs=100):
    y, _ = librosa.load(audio_path, sr=sr)
    # pitch estimation with pyin (may be slow)
    try:
        f0, voiced_flag, voiced_prob = librosa.pyin(
            y, fmin=librosa.note_to_hz('A0'), fmax=librosa.note_to_hz('C8'),
            sr=sr, hop_length=hop_length
        )
    except Exception:
        f0 = np.array([])
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=False)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length) if len(f0)>0 else np.array([0.0])
    if len(times) == 0:
        return np.zeros((1,88), dtype=np.float32), np.zeros((1,), dtype=np.float32), np.zeros((1,), dtype=np.float32)
    T = int(np.ceil(times[-1] * fs)) + 1
    piano_roll = np.zeros((T, 88), dtype=np.float32)
    onset_roll = np.zeros((T,), dtype=np.float32)
    duration_roll = np.zeros((T,), dtype=np.float32)
    for i, freq in enumerate(f0):
        if np.isfinite(freq):
            midi = int(round(librosa.hz_to_midi(freq)))
            idx = midi - 21
            if 0 <= idx < 88:
                t_idx = int(times[i] * fs)
                if t_idx < T:
                    piano_roll[t_idx, idx] = 1.0
    for of in onset_frames:
        t = int(librosa.frames_to_time(of, sr=sr, hop_length=hop_length) * fs)
        if t < T:
            onset_roll[t] = 1.0
    # crude durations: set duration at onset positions
    for p in range(88):
        active = piano_roll[:, p]
        i = 0
        while i < T:
            if active[i]:
                j = i + 1
                while j < T and active[j]:
                    j += 1
                dur = (j - i) / fs
                duration_roll[i] = dur
                i = j
            else:
                i += 1
    return piano_roll, onset_roll, duration_roll

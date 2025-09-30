import librosa
import torch
import torch.nn.functional as F
import numpy as np  

MAX_LEN = 256  # Longitud máxima de la secuencia de entrada 

def estimate_tempo(audio_path, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats

def audio_to_mel(audio, sr=16000, n_mels=229, hop_length=512, n_fft=2048):
    if isinstance(audio, str):
        y, sr = librosa.load(audio, sr=sr)
    else:
        y = audio
    S = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    # Normalización por canal (cada frecuencia)
    mean = np.mean(S_db, axis=1, keepdims=True)
    std = np.std(S_db, axis=1, keepdims=True)
    S_norm = (S_db - mean) / (std + 1e-6)

    return S_norm.T.astype(np.float32)  # (frames, n_mels) 

def collate_fn(batch):
    # batch es lista de tuples (mel, notes, durs)
    # cada uno mel: (T_i, n_mels), notes: (T_i, 88), durs: (T_i, 88)

    # max_len = max(sample[0].shape[0] for sample in batch)  # max T
    max_len = min(MAX_LEN, max(sample[0].shape[0] for sample in batch))
    mel_list, notes_list, durs_list = [], [], []

    for mel, notes, durs in batch:
        T, n_mels = mel.shape
        pad_len = max_len - T
        
        # Pad temporal dimension (dim=0) con ceros
        mel_padded = F.pad(mel, (0,0,0,pad_len), "constant", 0)
        notes_padded = F.pad(notes, (0,0,0,pad_len), "constant", 0)
        durs_padded = F.pad(durs, (0,0,0,pad_len), "constant", 0)

        # notes_down = notes_padded[::2]
        # durs_down = durs_padded[::2]


        mel_list.append(mel_padded)
        notes_list.append(notes_padded)
        durs_list.append(durs_padded)

    mel_batch = torch.stack(mel_list)       # (B, max_len, n_mels)
    notes_batch = torch.stack(notes_list)   # (B, max_len, 88)
    durs_batch = torch.stack(durs_list)     # (B, max_len, 88)

    return mel_batch, notes_batch, durs_batch

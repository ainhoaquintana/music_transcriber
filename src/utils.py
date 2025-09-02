import librosa
import torch
import torch.nn.functional as F
import numpy as np  

MAX_LEN = 256  # Longitud máxima de la secuencia de entrada 

def estimate_tempo(audio_path, sr=16000):
    y, _ = librosa.load(audio_path, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats

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

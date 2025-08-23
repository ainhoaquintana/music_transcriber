import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import librosa
import pretty_midi

MIN_MIDI = 21
MAX_MIDI = 108

class MaestroDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=229, hop_length=512):
    # def __init__(self, root_dir, sr=16000, n_mels=229, hop_length=256):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        # Buscar pares WAV-MIDI recursivamente
        wavs = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
        print(f"Audios: {len(wavs)}")
        self.pairs = []
        for w in wavs:
            base = os.path.splitext(w)[0]
            for ext in [".mid", ".midi"]:
                m = base + ext
                if os.path.exists(m):
                    self.pairs.append((w, m))
                    break
        if not self.pairs:
            raise RuntimeError(f"No se encontraron pares WAV-MIDI en {root_dir}")

    def __len__(self):
        return len(self.pairs)

    def _audio_to_mel(self, path):
        y, _ = librosa.load(path, sr=self.sr, mono=True)
        S = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length)
        S_db = librosa.power_to_db(S, ref=np.max)
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
        return S_norm.T.astype(np.float32)  # Transpuesta: (T, n_mels)

    def _midi_to_labels(self, path, n_frames):
        pm = pretty_midi.PrettyMIDI(path)
        frame_time = self.hop_length / self.sr
        notes = np.zeros((n_frames, 88), np.float32)
        durs = np.zeros((n_frames, 88), np.float32)

        for inst in pm.instruments:
            for note in inst.notes:
                p = note.pitch - MIN_MIDI
                if 0 <= p < 88:
                    start = int(np.floor(note.start / frame_time))
                    end = int(np.ceil(note.end / frame_time))
                    if start >= n_frames:
                        continue
                    end = min(end, n_frames - 1)
                    notes[start:end+1, p] = 1.0
                    durs[start, p] = note.end - note.start
        return notes, durs


    def __getitem__(self, idx):
        wav_path, midi_path = self.pairs[idx]
        
        mel = self._audio_to_mel(wav_path)  # (T, n_mels)
        n_frames = mel.shape[0]             # número de frames en tiempo
        
        notes, durs = self._midi_to_labels(midi_path, n_frames)  # (T, 88)
        
        return (
            torch.from_numpy(mel),         # (T, n_mels)
            torch.from_numpy(notes),       # (T, 88)
            torch.from_numpy(durs)         # (T, 88)
        )

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from utils import audio_to_mel
import librosa
import pretty_midi

MIN_MIDI = 21
MAX_MIDI = 108


class MaestroDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=229, hop_length=512):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

        wavs = sorted(glob.glob(os.path.join(root_dir, "**", "*.wav"), recursive=True))
        print(f"Audios Maestro encontrados en {root_dir}: {len(wavs)}")

        self.pairs = []
        for w in wavs:
            base = os.path.splitext(w)[0]
            midi_path = None
            for ext in [".mid", ".midi"]:
                m = base + ext
                if os.path.exists(m):
                    midi_path = m
                    break
            if midi_path is None:
                continue

            # Verificamos que el MIDI sea válido
            try:
                _ = pretty_midi.PrettyMIDI(midi_path)
                self.pairs.append((w, midi_path))
            except Exception as e:
                print(f"[WARNING] MIDI inválido en {midi_path}: {e}. Ignorando par.")

        if not self.pairs:
            raise RuntimeError(f"No se encontraron pares WAV-MIDI válidos en {root_dir}")

    def __len__(self):
        return len(self.pairs)

    # def _audio_to_mel(self, path):
    #     y, _ = librosa.load(path, sr=self.sr, mono=True)
    #     S = librosa.feature.melspectrogram(
    #         y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length
    #     )
    #     S_db = librosa.power_to_db(S, ref=np.max)
    #     S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    #     return S_norm.T.astype(np.float32)

    def _midi_to_labels(self, path, n_frames):
        pm = pretty_midi.PrettyMIDI(path)
        frame_time = self.hop_length / self.sr

        onsets = np.zeros((n_frames, 88), np.float32)
        frames = np.zeros((n_frames, 88), np.float32)

        for inst in pm.instruments:
            for note in inst.notes:
                p = note.pitch - MIN_MIDI
                if not (0 <= p < 88):
                    continue

                start = int(np.floor(note.start / frame_time))
                end = int(np.ceil(note.end / frame_time))

                if start >= n_frames:
                    continue

                end = min(end, n_frames - 1)

                # 🔥 onset solo en el primer frame
                onsets[start, p] = 1.0
                # 🔥 frames en todo el rango activo
                frames[start:end + 1, p] = 1.0

        return onsets, frames


    def __getitem__(self, idx):
        wav_path, midi_path = self.pairs[idx]
        mel = audio_to_mel(wav_path)
        # mel = self._audio_to_mel(wav_path)
        n_frames = mel.shape[0]
        onsets, frames = self._midi_to_labels(midi_path, n_frames)
        return (
            torch.from_numpy(mel),
            torch.from_numpy(onsets),
            torch.from_numpy(frames),
        )


class MusicNetDataset(Dataset):
    def __init__(self, root_dir, sr=16000, n_mels=229, hop_length=512):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length

        wavs = sorted(glob.glob(os.path.join(root_dir, "*.wav")))
        print(f"Audios MusicNet encontrados en {root_dir}: {len(wavs)}")

        self.pairs = []
        for w in wavs:
            base = os.path.splitext(os.path.basename(w))[0]
            midi_candidates = glob.glob(os.path.join(root_dir, f"{base}_*.mid"))
            if not midi_candidates:
                continue

            midi_path = midi_candidates[0]
            try:
                _ = pretty_midi.PrettyMIDI(midi_path)
                self.pairs.append((w, midi_path))
            except Exception as e:
                print(f"[WARNING] MIDI inválido en {midi_path}: {e}. Ignorando par.")

        if not self.pairs:
            raise RuntimeError(f"No se encontraron pares WAV-MIDI válidos en {root_dir}")

    def __len__(self):
        return len(self.pairs)

    # def _audio_to_mel(self, path):
    #     y, _ = librosa.load(path, sr=self.sr, mono=True)
    #     S = librosa.feature.melspectrogram(
    #         y=y, sr=self.sr, n_mels=self.n_mels, hop_length=self.hop_length
    #     )
    #     S_db = librosa.power_to_db(S, ref=np.max)
    #     S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    #     return S_norm.T.astype(np.float32)

    def _midi_to_labels(self, path, n_frames):
        pm = pretty_midi.PrettyMIDI(path)
        frame_time = self.hop_length / self.sr

        onsets = np.zeros((n_frames, 88), np.float32)
        frames = np.zeros((n_frames, 88), np.float32)

        for inst in pm.instruments:
            for note in inst.notes:
                p = note.pitch - MIN_MIDI
                if not (0 <= p < 88):
                    continue

                start = int(np.floor(note.start / frame_time))
                end = int(np.ceil(note.end / frame_time))

                if start >= n_frames:
                    continue

                end = min(end, n_frames - 1)

                # 🔥 onset solo en el primer frame
                onsets[start, p] = 1.0
                # 🔥 frames en todo el rango activo
                frames[start:end + 1, p] = 1.0

        return onsets, frames


    def __getitem__(self, idx):
        wav_path, midi_path = self.pairs[idx]
        # mel = self._audio_to_mel(wav_path)
        mel = audio_to_mel(wav_path)
        n_frames = mel.shape[0]
        onsets, frames = self._midi_to_labels(midi_path, n_frames)
        return (
            torch.from_numpy(mel),
            torch.from_numpy(onsets),
            torch.from_numpy(frames),
        )



class MyDataset(Dataset):
    def __init__(self, root_dir="../data/my_audios", sr=16000, n_mels=229, hop_length=512):
        self.root_dir = root_dir
        self.sr = sr
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.samples = []

        # Buscar pares wav/mid
        for file in os.listdir(root_dir):
            if file.endswith(".wav"):
                name = file[:-4]
                midi_path = os.path.join(root_dir, name + ".mid")
                wav_path = os.path.join(root_dir, file)
                if os.path.isfile(midi_path):
                    self.samples.append((wav_path, midi_path))

        if not self.samples:
            raise RuntimeError(f"No se encontraron pares WAV-MIDI válidos en {root_dir}")
        print(f"Audios sintéticos encontrados en {root_dir}: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _midi_to_labels(self, path, n_frames):
        pm = pretty_midi.PrettyMIDI(path)
        frame_time = self.hop_length / self.sr

        onsets = np.zeros((n_frames, 88), np.float32)
        frames = np.zeros((n_frames, 88), np.float32)

        for inst in pm.instruments:
            for note in inst.notes:
                p = note.pitch - MIN_MIDI
                if not (0 <= p < 88):
                    continue

                start = int(np.floor(note.start / frame_time))
                end = int(np.ceil(note.end / frame_time))
                if start >= n_frames:
                    continue
                end = min(end, n_frames - 1)

                onsets[start, p] = 1.0
                frames[start:end + 1, p] = 1.0

        return onsets, frames

    def __getitem__(self, idx):
        wav_path, midi_path = self.samples[idx]
        mel = audio_to_mel(wav_path)  # Convierte WAV a mel espectrogram
        n_frames = mel.shape[0]
        onsets, frames = self._midi_to_labels(midi_path, n_frames)
        return (
            torch.from_numpy(mel),
            torch.from_numpy(onsets),
            torch.from_numpy(frames),
        )
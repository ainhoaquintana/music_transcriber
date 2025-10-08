import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from model import CnnTransformerOnsetsFrames
import librosa
import numpy as np
import pretty_midi

WINDOW_SIZE = 10000
STRIDE = 8000
HOP_LENGTH = 256
SR = 16000
THRESHOLD = 0.5

def audio_to_mel(audio_path, sr=SR, n_mels=229, hop_length=HOP_LENGTH):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.T.astype(np.float32)

def infer_with_sliding_window(model, mel, device, window_size=WINDOW_SIZE, stride=STRIDE):
    if isinstance(mel, np.ndarray):
        mel = torch.tensor(mel, dtype=torch.float32)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)  # (1, T, n_mels)

    T = mel.shape[1]
    onsets_list, frames_list = []

    for start in range(0, T, stride):
        end = min(start + window_size, T)
        mel_chunk = mel[:, start:end, :].to(device)
        with torch.no_grad():
            onsets_logits, frames_logits = model(mel_chunk)
        onsets_list.append(onsets_logits.cpu())
        frames_list.append(frames_logits.cpu())
        if end == T:
            break

    onsets_full = torch.cat(onsets_list, dim=1)
    frames_full = torch.cat(frames_list, dim=1)
    return onsets_full[0], frames_full[0]

def infer_notes_from_onsets_frames(model, mel, device, threshold=THRESHOLD):
    onsets_logits, frames_logits = infer_with_sliding_window(model, mel, device)
    onsets_probs = torch.sigmoid(onsets_logits)
    frames_probs = torch.sigmoid(frames_logits)

    onsets_bin = (onsets_probs > threshold).int()
    frames_bin = (frames_probs > threshold).int()

    T, N = onsets_bin.shape
    notes_bin = torch.zeros_like(frames_bin)
    active_notes = {}

    for t in range(T):
        for n in range(N):
            if onsets_bin[t, n]:
                notes_bin[t, n] = 1
                active_notes[n] = t
            elif n in active_notes and frames_bin[t, n]:
                notes_bin[t, n] = 1
            elif n in active_notes and not frames_bin[t, n]:
                active_notes.pop(n)

    return notes_bin, onsets_probs, frames_probs

def notes_to_midi(notes_bin, output_path="debug_test.midi", sr=SR, hop_length=HOP_LENGTH):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    T, N = notes_bin.shape
    frame_duration = hop_length / sr
    active_notes = {}

    for t in range(T):
        for n in range(N):
            if notes_bin[t, n]:
                if n not in active_notes:
                    active_notes[n] = t
            else:
                if n in active_notes:
                    start_frame = active_notes.pop(n)
                    duration = (t - start_frame) * frame_duration
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=n + 21,
                        start=start_frame * frame_duration,
                        end=start_frame * frame_duration + duration
                    )
                    piano.notes.append(note)

    for n, start_frame in active_notes.items():
        duration = (T - start_frame) * frame_duration
        note = pretty_midi.Note(
            velocity=100,
            pitch=n + 21,
            start=start_frame * frame_duration,
            end=start_frame * frame_duration + duration
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"MIDI saved to {output_path}")

def main(audio_path):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join("outputs", audio_name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_entrenado.pth", map_location=device))
    model.to(device)
    model.eval()

    mel = audio_to_mel(audio_path)
    notes_bin, onsets_probs, frames_probs = infer_notes_from_onsets_frames(model, mel, device)

    plt.figure(figsize=(12,6))
    plt.subplot(3,1,1)
    plt.imshow(onsets_probs.cpu().T, aspect='auto', origin='lower', cmap='Reds')
    plt.title("Onsets probabilities")
    plt.ylabel("MIDI note")

    plt.subplot(3,1,2)
    plt.imshow(frames_probs.cpu().T, aspect='auto', origin='lower', cmap='Blues')
    plt.title("Frames probabilities")
    plt.ylabel("MIDI note")

    plt.subplot(3,1,3)
    plt.imshow(notes_bin.cpu().T, aspect='auto', origin='lower', cmap='Greens')
    plt.title("Notes reconstructed")
    plt.ylabel("MIDI note")
    plt.xlabel("Time frames")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_plot.png"))
    plt.show()

    # Guardar MIDI
    notes_to_midi(notes_bin, os.path.join(output_dir, "debug_test.midi"))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_onsets_frames.py path/to/audio")
        sys.exit(1)
    main(sys.argv[1])

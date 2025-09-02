import sys
import os
import torch
import torch.nn.functional as F
import pretty_midi
from music21 import converter, metadata, instrument
from model import CnnTransformerTranscriber
import librosa
import numpy as np

WINDOW_SIZE = 10000
STRIDE = 8000
HOP_LENGTH = 256
SR = 16000

def infer_with_sliding_window(model, mel, device, window_size=WINDOW_SIZE, stride=STRIDE):
    """
    Run inference on long spectrograms by splitting into overlapping windows.
    Returns logits and durations for the whole sequence.
    """
    if isinstance(mel, np.ndarray):
        mel = torch.tensor(mel, dtype=torch.float32)

    if mel.ndim == 2:
        mel = mel.unsqueeze(0)  # (1, T, n_mels)

    T = mel.shape[1]
    notes_list, durs_list = [], []

    for start in range(0, T, stride):
        end = min(start + window_size, T)
        mel_chunk = mel[:, start:end, :].to(device)
        with torch.no_grad():
            notes_logits, durs = model(mel_chunk)
            durs = F.relu(durs)  # avoid negative durations

        notes_list.append(notes_logits.cpu())
        durs_list.append(durs.cpu())

        if end == T:
            break

    notes_logits_full = torch.cat(notes_list, dim=1)
    durs_full = torch.cat(durs_list, dim=1)
    return notes_logits_full[0], durs_full[0]  # remove batch dimension


def infer_notes_and_durations(model, mel, device):
    notes_logits, dur_preds = infer_with_sliding_window(model, mel, device)
    notes_probs = torch.sigmoid(notes_logits)
    notes_bin = (notes_probs > 0.5).int()

    print("notes_logits stats:", notes_logits.min().item(), notes_logits.max().item(), notes_logits.mean().item())
    print("notes_probs stats:", notes_probs.min().item(), notes_probs.max().item(), notes_probs.mean().item())
    print("notes_bin sum:", notes_bin.sum().item())
    print("durations stats:", dur_preds.min().item(), dur_preds.max().item(), dur_preds.mean().item())

    return notes_bin, dur_preds


def notes_to_midi(notes_bin, durations, output_path="output.midi", sr=SR, hop_length=HOP_LENGTH):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    T, N = notes_bin.shape
    frame_duration = hop_length / sr  # duration of one frame in seconds

    # Track active notes to accumulate duration
    active_notes = dict()

    for t in range(T):
        for n in range(N):
            if notes_bin[t, n]:
                if n not in active_notes:
                    # Start new note
                    active_notes[n] = t
            else:
                if n in active_notes:
                    # End note
                    start_frame = active_notes.pop(n)
                    duration = (t - start_frame) * frame_duration
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=n + 21,
                        start=start_frame * frame_duration,
                        end=start_frame * frame_duration + duration
                    )
                    piano.notes.append(note)

    # Any notes still active at the end
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


def midi_to_musicxml(midi_path, xml_path, audio_name):
    score = converter.parse(midi_path)
    score.metadata = metadata.Metadata()
    score.metadata.title = audio_name
    score.metadata.composer = "Your Name"

    for part in score.parts:
        part.insert(0, instrument.Piano())
        part.partName = "Piano"

    score.write('musicxml', xml_path)
    print(f"MusicXML saved to {xml_path}")


def audio_to_mel(audio_path, sr=SR, n_mels=229, hop_length=HOP_LENGTH):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.T.astype(np.float32)  # (T, n_mels)


def main(audio_path):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join("outputs", audio_name)
    os.makedirs(output_dir, exist_ok=True)
    midi_path = os.path.join(output_dir, f"{audio_name}.midi")
    xml_path = os.path.join(output_dir, f"{audio_name}.xml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnTransformerTranscriber(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_entrenado.pth", map_location=device))
    model.to(device)
    model.eval()

    mel = audio_to_mel(audio_path)
    notes_bin, durations = infer_notes_and_durations(model, mel, device)
    notes_to_midi(notes_bin, durations, midi_path, sr=SR, hop_length=HOP_LENGTH)
    midi_to_musicxml(midi_path, xml_path, audio_name)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcription.py path/to/audio")
        sys.exit(1)
    main(sys.argv[1])

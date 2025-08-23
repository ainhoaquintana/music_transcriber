import sys
import os
import torch
import pretty_midi
from music21 import converter, metadata, instrument
from model import CnnTransformerTranscriber
import librosa
import numpy as np

def audio_to_mel(audio_path, sr=16000, n_mels=229, hop_length=512):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.T.astype(np.float32)  # (T, n_mels)

def infer_notes_and_durations(model, mel, device, threshold=0.5):
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, n_mels)
    with torch.no_grad():
        notes_logits, dur_preds = model(mel)
        notes_probs = torch.sigmoid(notes_logits)  # (1, T_reduced, n_notes)
        notes_bin = (notes_probs > threshold).cpu().numpy()[0]  # binary notes
        durations = dur_preds.cpu().numpy()[0]                  # predicted durations
        print("notes_logits stats:", notes_logits.min().item(), notes_logits.max().item(), notes_logits.mean().item())
        print("notes_probs stats:", notes_probs.min().item(), notes_probs.max().item(), notes_probs.mean().item())
        print("notes_bin sum:", notes_bin.sum())
    return notes_bin, durations

def notes_to_midi(notes_bin, durations, output_path="output.midi"):
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    time = 0.0
    for t in range(notes_bin.shape[0]):
        for note_num in range(notes_bin.shape[1]):
            if notes_bin[t, note_num]:
                duration = max(durations[t, note_num], 0.05)
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_num + 21,
                    start=time,
                    end=time + duration
                )
                piano.notes.append(note)
        time += 0.05  # or your frame hop duration
    midi.instruments.append(piano)
    midi.write(output_path)

def midi_to_musicxml(midi_path, xml_path, audio_name):
    score = converter.parse(midi_path)
    score.metadata = metadata.Metadata()
    score.metadata.title = audio_name
    score.metadata.composer = "TFM Model"
    for part in score.parts:
        part.insert(0, instrument.Piano())
        part.partName = "Piano"
    score.write('musicxml', xml_path)

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
    notes_bin, durations = infer_notes_and_durations(model, mel, device, threshold=0.05)
    notes_to_midi(notes_bin, durations, midi_path)
    midi_to_musicxml(midi_path, xml_path, audio_name)
    print(f"Saved MIDI to {midi_path}")
    print(f"Saved MusicXML to {xml_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python infer_to_xml.py path/to/audio.wav")
        sys.exit(1)
    main(sys.argv[1])
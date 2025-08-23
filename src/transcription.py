import sys
import os
import torch
import torch.nn.functional as F
import pretty_midi
from music21 import converter, metadata, instrument
from model import CnnTransformerTranscriber
import librosa
import matplotlib.pyplot as plt
import numpy as np

# Función para inferir notas y duraciones a partir de un espectrograma mel
def infer_notes_and_durations(model, mel, device):
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).to(device)  # (1, T, n_mels)
    with torch.no_grad():
        notes_logits, dur_preds = model(mel)
        notes_probs = torch.sigmoid(notes_logits)  # (1, T_reduced, n_notes)
        notes_bin = (notes_probs > 0.1).cpu().numpy()[0]  # binary notes
        durations = dur_preds.cpu().numpy()[0]            # predicted durations
        # print("mel shape:", mel.shape)
        print("notes_logits stats:", notes_logits.min().item(), notes_logits.max().item(), notes_logits.mean().item())
        print("notes_probs stats:", notes_probs.min().item(), notes_probs.max().item(), notes_probs.mean().item())
        print("notes_bin sum:", notes_bin.sum())
    return notes_bin, durations

def notes_to_midi(notes_bin, durations, output_path="output.midi"):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    # print(f"Converting {notes_bin.shape[0]} frames to MIDI...")
    # print(notes_bin)
    time = 0.0
    for t in range(notes_bin.shape[0]):
        for note_num in range(notes_bin.shape[1]):
            if notes_bin[t, note_num]:
                duration = max(durations[t, note_num], 0.05)  # avoid zero duration
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=note_num + 21,  # MIDI note numbers (A0=21)
                    start=time,
                    end=time + duration
                )
                instrument.notes.append(note)
        time += 0.05  # or your frame hop duration
    midi.instruments.append(instrument)
    midi.write(output_path)
   
    # pm = pretty_midi.PrettyMIDI("../data/maestro-v3.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi")
    # piano_roll = pm.get_piano_roll(fs=100)  # fs = frames per second

    # plt.figure(figsize=(12, 6))
    # plt.imshow(piano_roll, aspect='auto', origin='lower', cmap='hot')
    # plt.xlabel("Time (frames)")
    # plt.ylabel("MIDI pitch")
    # plt.title("Piano Roll")
    # plt.colorbar(label="Velocity")
    # plt.show()


def midi_to_musicxml(midi_path, xml_path, audio_name):
    score = converter.parse(midi_path)
    # Add metadata
    score.metadata = metadata.Metadata()
    score.metadata.title = audio_name
    score.metadata.composer = "Your Name"
    # Set instrument name
    for part in score.parts:
        part.insert(0, instrument.Piano())  # or instrument.Instrument('Violin'), etc.
        part.partName = "Piano"
    score.write('musicxml', xml_path)

def audio_to_mel(audio_path, sr=16000, n_mels=229, hop_length=512):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.T.astype(np.float32)  # Transpuesta: (T, n_mels)

def main(audio_path):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join("outputs", audio_name)
    os.makedirs(output_dir, exist_ok=True)
    midi_path = os.path.join(output_dir, f"{audio_name}.midi")
    xml_path = os.path.join(output_dir, f"{audio_name}.xml")


    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnTransformerTranscriber(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_entrenado.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load your mel spectrogram here (replace with your actual loading code)
    mel = audio_to_mel(audio_path)
    # print("hola mel shape:", mel.shape)
    plt.figure(figsize=(12, 6))
    plt.imshow(mel, aspect='auto', origin='lower', cmap='hot')
    plt.xlabel("mel frames")
    plt.ylabel("MIDI pitch")
    plt.title("Piano Roll")
    plt.colorbar(label="Velocity")
    plt.show()

    # Infer notes and durations
    notes_bin, durations = infer_notes_and_durations(model, mel, device)
    notes_to_midi(notes_bin, durations, midi_path)
    midi_to_musicxml(midi_path, xml_path, audio_name)
    print(f"Saved MIDI to {midi_path}")
    print(f"Saved MusicXML to {xml_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcription.py path/to/audio")
        sys.exit(1)
    main(sys.argv[1])
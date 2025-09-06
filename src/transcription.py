import sys
import os
import torch
import torch.nn.functional as F
import pretty_midi
from music21 import converter, metadata, instrument, tempo
from model import CnnTransformerOnsetsFrames
import librosa
import numpy as np

WINDOW_SIZE = 10000
STRIDE = 8000
HOP_LENGTH = 256
SR = 16000

def infer_with_sliding_window(model, mel, device, window_size=WINDOW_SIZE, stride=STRIDE):
    """Run inference on long spectrograms in overlapping windows."""
    if isinstance(mel, np.ndarray):
        mel = torch.tensor(mel, dtype=torch.float32)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)  # (1, T, n_mels)

    T = mel.shape[1]
    onsets_list, frames_list = [], []

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

def infer_notes_from_onsets_frames(model, mel, device, threshold=0.5, median_window=3):
    """Infer notes using Onsets and Frames with median filtering to remove spurious notes."""
    onsets_logits, frames_logits = infer_with_sliding_window(model, mel, device)
    onsets_bin = (torch.sigmoid(onsets_logits) > threshold).int()
    frames_bin = (torch.sigmoid(frames_logits) > threshold).int()

    # Median filter along time dimension
    pad = median_window // 2
    frames_bin_padded = F.pad(frames_bin.T.unsqueeze(0).float(), (pad, pad), mode='replicate')
    frames_smooth = frames_bin_padded.unfold(2, median_window, 1).median(dim=3).values.squeeze(0).T.int()

    T, N = onsets_bin.shape
    notes_bin = torch.zeros_like(frames_smooth)
    active_notes = {}

    for t in range(T):
        for n in range(N):
            if onsets_bin[t, n]:
                notes_bin[t, n] = 1
                active_notes[n] = t
            elif n in active_notes and frames_smooth[t, n]:
                notes_bin[t, n] = 1
            elif n in active_notes and not frames_smooth[t, n]:
                active_notes.pop(n)
    return notes_bin

def audio_to_mel(audio_path, sr=SR, n_mels=229, hop_length=HOP_LENGTH):
    y, _ = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.T.astype(np.float32)

def detect_tempo(audio_path, sr=SR):
    y, _ = librosa.load(audio_path, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return tempo

def round_durations_to_grid(durations_sec, tempo, beat_unit=0.25):
    """
    Round durations to nearest rhythmic value (quarter, eighth, etc.)
    beat_unit: fraction of whole note (0.25 = quarter note)
    """
    # duration of a quarter note in seconds
    quarter_sec = 60.0 / tempo
    grid = np.array([1, 0.5, 0.25, 0.125, 0.0625]) * quarter_sec
    rounded = np.array([grid[np.argmin(np.abs(grid - d))] for d in durations_sec])
    return rounded

def notes_to_midi(notes_bin, output_path="output.midi", sr=SR, hop_length=HOP_LENGTH, tempo_bpm=120, min_duration_sec=0.05):
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
                    duration = max(duration, min_duration_sec)
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=int(n) + 21,
                        start=start_frame * frame_duration,
                        end=start_frame * frame_duration + duration
                    )
                    piano.notes.append(note)

    for n, start_frame in active_notes.items():
        duration = (T - start_frame) * frame_duration
        duration = max(duration, min_duration_sec)
        note = pretty_midi.Note(
            velocity=100,
            pitch=int(n) + 21,
            start=start_frame * frame_duration,
            end=start_frame * frame_duration + duration
        )
        piano.notes.append(note)

    midi.instruments.append(piano)
    midi.estimate_tempo()  # optional, but we'll set explicit tempo below
    # set tempo
    midi._PrettyMIDI__tick_scales = {0: 60.0 / tempo_bpm}  # hack para tempo
    midi.write(output_path)
    print(f"MIDI saved to {output_path}")

def midi_to_musicxml(midi_path, xml_path, audio_name, tempo_bpm=120):
    score = converter.parse(midi_path)
    score.metadata = metadata.Metadata()
    score.metadata.title = audio_name
    score.metadata.composer = "Your Name"

    # Insert tempo marking
    mm = tempo.MetronomeMark(number=tempo_bpm)
    score.insert(0, mm)

    for part in score.parts:
        part.insert(0, instrument.Piano())
        part.partName = "Piano"

    score.write('musicxml', xml_path)
    print(f"MusicXML saved to {xml_path}")

def main(audio_path):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join("outputs", audio_name)
    os.makedirs(output_dir, exist_ok=True)
    midi_path = os.path.join(output_dir, f"{audio_name}.midi")
    xml_path = os.path.join(output_dir, f"{audio_name}.xml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_entrenado.pth", map_location=device))
    model.to(device)
    model.eval()

    mel = audio_to_mel(audio_path)
    tempo_bpm = detect_tempo(audio_path)
    notes_bin = infer_notes_from_onsets_frames(model, mel, device, threshold=0.9, median_window=9)
    notes_to_midi(notes_bin, midi_path, sr=SR, hop_length=HOP_LENGTH, tempo_bpm=tempo_bpm)
    midi_to_musicxml(midi_path, xml_path, audio_name, tempo_bpm=tempo_bpm)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcription.py path/to/audio")
        sys.exit(1)
    main(sys.argv[1])

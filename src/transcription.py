# transcription.py
import sys
import os
import torch
import torch.nn.functional as F
import pretty_midi
from music21 import converter, metadata, instrument, tempo
from model import CnnTransformerOnsetsFrames
import librosa
from utils import audio_to_mel
import numpy as np

# ---------------- Configuración global ----------------
SR = 16000
HOP_LENGTH = 512
MIN_DURATION_SEC = 0.15
MIN_FRAMES = 5
PITCH_OFFSET = 5

# Sliding window
WINDOW_SIZE = 20000
STRIDE = 15000

# Post-procesado
MEDIAN_WINDOW = 9
ONSET_MULTIPLIER = 1.0  # ajustado para detectar onsets
FRAME_MULTIPLIER = 1.0  # ajustado para frames
FRAME_OFF_HYST = 3
TOPK = 0

# ---------------- Funciones de inferencia ----------------
def infer_with_sliding_window(model, mel, device, window_size=WINDOW_SIZE, stride=STRIDE):
    if isinstance(mel, np.ndarray):
        mel = torch.tensor(mel, dtype=torch.float32)
    if mel.ndim == 2:
        mel = mel.unsqueeze(0)

    model = model.to(device)
    model.eval()

    T = mel.shape[1]
    onsets_list, frames_list = [], []

    for start in range(0, T, stride):
        end = min(start + window_size, T)
        mel_chunk = mel[:, start:end, :].to(device)
        with torch.no_grad():
            onsets_logits_chunk, frames_logits_chunk = model(mel_chunk)
        onsets_list.append(onsets_logits_chunk.cpu())
        frames_list.append(frames_logits_chunk.cpu())
        if end == T:
            break

    onsets_full = torch.cat(onsets_list, dim=1)[0]
    frames_full = torch.cat(frames_list, dim=1)[0]
    return onsets_full, frames_full

# ---------------- Post-procesado ----------------
def median_smooth_probabilities(prob, k):
    if k <= 1:
        return prob
    pad = k // 2
    prob_padded = np.pad(prob, ((pad, pad), (0, 0)), mode='edge')
    T = prob.shape[0]
    sm = np.zeros_like(prob)
    for t in range(T):
        sm[t] = np.median(prob_padded[t:t + k, :], axis=0)
    return sm

def onset_peak_picking(onset_probs, onset_thresh, width=0):
    T, N = onset_probs.shape
    peaks = np.zeros_like(onset_probs, dtype=np.bool_)
    for n in range(N):
        col = onset_probs[:, n]
        for t in range(T):
            if col[t] <= onset_thresh:
                continue
            left = col[max(0, t - width):t]
            right = col[t + 1:t + 1 + width]
            if (left.size == 0 or col[t] >= left.max()) and (right.size == 0 or col[t] >= right.max()):
                peaks[t, n] = True
    return peaks

def notes_from_probs(onset_probs, frame_probs, hop_length=HOP_LENGTH, sr=SR,
                     onset_multiplier=ONSET_MULTIPLIER, frame_multiplier=FRAME_MULTIPLIER,
                     median_window=MEDIAN_WINDOW, min_duration_sec=MIN_DURATION_SEC,
                     min_frames=MIN_FRAMES, frame_off_hyst=FRAME_OFF_HYST, topk=TOPK):

    on_p = onset_probs.copy()
    fr_p = frame_probs.copy()
    fr_p = median_smooth_probabilities(fr_p, median_window)

    onset_thresh = max(0.01, on_p.mean()) * onset_multiplier
    frame_thresh = max(0.01, fr_p.mean()) * frame_multiplier

    onset_peaks = onset_peak_picking(on_p, onset_thresh)
    print("Num detected onsets:", onset_peaks.sum())

    T, N = on_p.shape
    notes_bin = np.zeros((T, N), dtype=np.int32)

    active = {}
    for t in range(T):
        if topk > 0:
            topk_idx = np.argsort(fr_p[t])[::-1][:topk]
            frame_mask = np.zeros(N, dtype=bool)
            frame_mask[topk_idx] = True
        else:
            frame_mask = fr_p[t] >= frame_thresh

        for n in range(N):
            if onset_peaks[t, n] and frame_mask[n]:
                if n not in active:
                    active[n] = (t, 0)
                notes_bin[t, n] = 1
            elif n in active:
                s, below = active[n]
                if frame_mask[n]:
                    active[n] = (s, 0)
                    notes_bin[t, n] = 1
                else:
                    below += 1
                    if below >= frame_off_hyst:
                        active.pop(n)
                    else:
                        active[n] = (s, below)
                        notes_bin[t, n] = 1

    # filtrar por duración mínima
    frame_dur = hop_length / sr
    final_roll = np.zeros_like(notes_bin)
    active = {}
    for t in range(T):
        for n in range(N):
            if notes_bin[t, n]:
                if n not in active:
                    active[n] = t
            else:
                if n in active:
                    s = active.pop(n)
                    dur_frames = t - s
                    dur_sec = dur_frames * frame_dur
                    if dur_sec >= min_duration_sec and dur_frames >= min_frames:
                        final_roll[s:t, n] = 1
    for n, s in active.items():
        dur_frames = T - s
        dur_sec = dur_frames * frame_dur
        if dur_sec >= min_duration_sec and dur_frames >= min_frames:
            final_roll[s:T, n] = 1
    return final_roll

# ---------------- I/O ----------------
def detect_tempo(audio_path, sr=SR):
    y, _ = librosa.load(audio_path, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return float(tempo)

def notes_to_midi(notes_bin, output_path="output.midi", sr=SR, hop_length=HOP_LENGTH,
                  tempo_bpm=120, min_duration_sec=MIN_DURATION_SEC, max_duration_sec=2.0, min_frames=MIN_FRAMES):

    if isinstance(notes_bin, torch.Tensor):
        notes_bin = notes_bin.cpu().numpy().astype(np.int32)
    T, N = notes_bin.shape
    frame_duration = float(hop_length) / float(sr)

    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    piano = pretty_midi.Instrument(program=0)
    active = {}
    for t in range(T):
        for n in range(N):
            if notes_bin[t, n]:
                if n not in active:
                    active[n] = t
            else:
                if n in active:
                    s = active.pop(n)
                    dur_frames = t - s
                    dur_sec = dur_frames * frame_duration
                    if dur_sec >= min_duration_sec and dur_frames >= min_frames:
                        dur_sec = min(dur_sec, max_duration_sec)
                        note = pretty_midi.Note(
                            velocity=100,
                            pitch=int(n) + 21 + PITCH_OFFSET,
                            start=float(s) * frame_duration,
                            end=float(s) * frame_duration + float(dur_sec)
                        )
                        piano.notes.append(note)
    for n, s in active.items():
        dur_frames = T - s
        dur_sec = dur_frames * frame_duration
        if dur_sec >= min_duration_sec and dur_frames >= min_frames:
            dur_sec = min(dur_sec, max_duration_sec)
            note = pretty_midi.Note(
                velocity=100,
                pitch=int(n) + 21 + PITCH_OFFSET,
                start=float(s) * frame_duration,
                end=float(s) * frame_duration + float(dur_sec)
            )
            piano.notes.append(note)

    midi.instruments.append(piano)
    midi.write(output_path)
    print(f"MIDI saved to {output_path} with tempo {tempo_bpm} BPM")

def midi_to_musicxml(midi_path, xml_path, audio_name, tempo_bpm=120):
    score = converter.parse(midi_path)
    score.metadata = metadata.Metadata()
    score.metadata.title = audio_name
    score.metadata.composer = "Auto Transcriber"
    mm = tempo.MetronomeMark(number=tempo_bpm)
    score.insert(0, mm)
    for part in score.parts:
        part.insert(0, instrument.Piano())
        part.partName = "Piano"
    score.write('musicxml', xml_path)
    print(f"MusicXML saved to {xml_path}")

# ---------------- Main ----------------
def main(audio_path):
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = os.path.join("outputs", audio_name)
    os.makedirs(output_dir, exist_ok=True)
    midi_path = os.path.join(output_dir, f"{audio_name}.midi")
    xml_path = os.path.join(output_dir, f"{audio_name}.xml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_final_focal.pth", map_location=device))
    # model.load_state_dict(torch.load("../checkpoints/modelo_finetuned_monophonic.pth", map_location=device))
    model.to(device)
    model.eval()

    mel = audio_to_mel(audio_path)
    tempo_bpm = detect_tempo(audio_path)

    onsets_logits, frames_logits = infer_with_sliding_window(model, mel, device)
    onsets_probs = torch.sigmoid(onsets_logits).cpu().numpy()
    frames_probs = torch.sigmoid(frames_logits).cpu().numpy()

    print("📊 Mean onset prob:", onsets_probs.mean())
    print("📊 Max onset prob:", onsets_probs.max())
    print("📊 Mean frame prob:", frames_probs.mean())
    print("📊 Max frame prob:", frames_probs.max())

    notes_roll = notes_from_probs(onsets_probs, frames_probs)

    notes_to_midi(notes_roll, midi_path, sr=SR, hop_length=HOP_LENGTH, tempo_bpm=tempo_bpm)
    midi_to_musicxml(midi_path, xml_path, audio_name, tempo_bpm=tempo_bpm)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcription.py path/to/audio")
        sys.exit(1)
    main(sys.argv[1])

# audio_create.py
import os
import numpy as np
import pretty_midi
import soundfile as sf

# Carpeta de salida
OUTPUT_DIR = "../data/my_audios"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Carpeta donde están los soundfonts
SOUNDFONT_DIR = "../soundfonts"
SOUNDFONT_FILE = "GeneralUser-GS.sf2" 
SOUNDFONT_PATH = os.path.join(SOUNDFONT_DIR, SOUNDFONT_FILE)

# Parámetros
N_MELODIES = 300
MIN_NOTES = 5
MAX_NOTES = 15
MIN_PITCH = 60   # C4
MAX_PITCH = 72   # C5
MIN_DURATION = 0.3
MAX_DURATION = 1.0
SAMPLE_RATE = 16000

def generate_random_melody():
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)  # Piano acústico
    current_time = 0.0

    n_notes = np.random.randint(MIN_NOTES, MAX_NOTES + 1)
    for _ in range(n_notes):
        pitch = np.random.randint(MIN_PITCH, MAX_PITCH + 1)
        duration = np.random.uniform(MIN_DURATION, MAX_DURATION)
        note = pretty_midi.Note(
            velocity=100,
            pitch=pitch,
            start=current_time,
            end=current_time + duration
        )
        piano.notes.append(note)
        current_time += duration  # monofonía secuencial

    pm.instruments.append(piano)
    return pm

def midi_to_wav(pm, wav_path, sf2_path=SOUNDFONT_PATH, sr=SAMPLE_RATE):
    if not os.path.isfile(sf2_path):
        raise FileNotFoundError(f"Soundfont no encontrado: {sf2_path}")
    audio = pm.fluidsynth(fs=sr, sf2_path=sf2_path)
    sf.write(wav_path, audio, sr)

def main():
    for i in range(N_MELODIES):
        name = f"melody_{i:03d}"
        midi_path = os.path.join(OUTPUT_DIR, name + ".mid")
        wav_path = os.path.join(OUTPUT_DIR, name + ".wav")

        pm = generate_random_melody()
        pm.write(midi_path)
        print(f"Saved MIDI: {midi_path}")

        try:
            midi_to_wav(pm, wav_path)
            print(f"Saved WAV: {wav_path}")
        except Exception as e:
            print(f"Error generating WAV for {name}: {e}")

if __name__ == "__main__":
    main()

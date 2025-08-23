import numpy as np
import soundfile as sf
import os

# Crear carpeta si no existe
os.makedirs("data/test_audio", exist_ok=True)

# Parámetros generales
sr = 22050  # Frecuencia de muestreo
note_duration = 0.5  # Duración de cada nota en segundos
volume = 0.5  # Volumen

# Notas (frecuencias en Hz) - C4, D4, E4, F4, G4, A4, B4, C5
notes_freq = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]

# Crear melodía: Do-Re-Mi-Fa-Mi-Re-Do
melody_sequence = [0, 1, 2, 3, 2, 1, 0]

# Generar audio concatenando cada nota
melody = np.array([])
for note_index in melody_sequence:
    t = np.linspace(0, note_duration, int(sr * note_duration), endpoint=False)
    y = volume * np.sin(2 * np.pi * notes_freq[note_index] * t)
    melody = np.concatenate((melody, y))

# Guardar en WAV
output_path = "data/test_audio/test_melody.wav"
sf.write(output_path, melody, sr)

print(f"Melodía guardada en: {output_path}")

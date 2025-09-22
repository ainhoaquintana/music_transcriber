import torch
import torch.nn.functional as F
import numpy as np
import librosa
from model import CnnTransformerOnsetsFrames

# ------------------- CONFIG -------------------
SR = 16000          # sample rate
DURATION = 2.0      # duración en segundos
FREQ = 440.0        # frecuencia del test (Hz)
N_MELS = 229
HOP_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../checkpoints/modelo_entrenado.pth"
# ----------------------------------------------

# 1️⃣ Generar seno puro
t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)
y = 0.5 * np.sin(2 * np.pi * FREQ * t)

# 2️⃣ Convertir a mel-spectrogram
S = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH)
S_db = librosa.power_to_db(S, ref=np.max)
S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
mel_input = torch.tensor(S_norm.T, dtype=torch.float32).unsqueeze(0)  # (1, T, n_mels)

# 3️⃣ Cargar modelo
model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# 4️⃣ Inferencia
with torch.no_grad():
    onsets_logits, frames_logits = model(mel_input.to(DEVICE))

frames_probs = torch.sigmoid(frames_logits[0])  # (T, 88)

# 5️⃣ Ver nota más probable por frame
topk_vals, topk_idx = torch.topk(frames_probs, k=1, dim=-1)  # nota más probable por frame
topk_idx_flat = topk_idx.view(-1)
most_frequent_note = torch.mode(topk_idx_flat).values.item()  # ahora sí da un escalar

print(f"Frecuencia de prueba: {FREQ} Hz")
print(f"Nota MIDI más frecuente predicha: {most_frequent_note + 21}")
print(f"Nota de piano estimada (ej: La4 = 69): {most_frequent_note + 21}")

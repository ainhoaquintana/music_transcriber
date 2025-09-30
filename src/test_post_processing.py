import sys
import torch
import librosa
import numpy as np
import pandas as pd
from model import CnnTransformerOnsetsFrames

# Parámetros globales
SR = 16000
HOP_LENGTH = 256
N_MELS = 229
N_NOTES = 88  # MIDI 21 - 108

def audio_to_mel_array(y, sr=SR, n_mels=N_MELS, hop_length=HOP_LENGTH):
    """Convierte audio en un mel-espectrograma normalizado (frames x mels)."""
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)
    return S_norm.T.astype(np.float32)


def main():
    if len(sys.argv) < 2:
        print("Uso: python test_postprocessing.py <ruta_audio.wav> [salida.csv]")
        sys.exit(1)

    audio_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "salida.csv"

    y, sr = librosa.load(audio_path, sr=SR)
    print(f"✅ Audio cargado: {audio_path} ({len(y)/sr:.2f}s, {sr} Hz)")

    mel = audio_to_mel_array(y)
    T = mel.shape[0]
    print(f"✅ Mel espectrograma: {mel.shape} (frames x mels)")

    # 3️⃣ Cargar modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnTransformerOnsetsFrames(d_model=512, n_mels=N_MELS, n_notes=N_NOTES)
    model.load_state_dict(torch.load("../checkpoints/modelo_entrenado_sin_fine_tuning.pth", map_location=device))
    model.to(device)
    model.eval()

    # 4️⃣ Inferencia
    mel_t = torch.tensor(mel).unsqueeze(0).to(device)  # (1, T, n_mels)
    with torch.no_grad():
        _, frames_logits = model(mel_t)
        probs = torch.sigmoid(frames_logits[0])  # (T, 88)

    times = librosa.frames_to_time(np.arange(T), sr=SR, hop_length=HOP_LENGTH)

    data = {"Time(s)": times}
    for idx, midi in enumerate(range(21, 109)):
        data[f"Note_{midi}"] = probs[:, idx].cpu().numpy()

    df = pd.DataFrame(data)

    top1_list, top2_list, top3_list = [], [], []
    top1_prob, top2_prob, top3_prob = [], [], []

    probs_np = probs.cpu().numpy()
    for t in range(T):
        top_idx = np.argsort(probs_np[t])[::-1][:3]  # índices de las 3 más altas
        top_probs = probs_np[t][top_idx]
        top_midis = [21 + i for i in top_idx]

        top1_list.append(top_midis[0])
        top2_list.append(top_midis[1])
        top3_list.append(top_midis[2])
        top1_prob.append(top_probs[0])
        top2_prob.append(top_probs[1])
        top3_prob.append(top_probs[2])

    df["Top1_MIDI"] = top1_list
    df["Top1_Prob"] = top1_prob
    df["Top2_MIDI"] = top2_list
    df["Top2_Prob"] = top2_prob
    df["Top3_MIDI"] = top3_list
    df["Top3_Prob"] = top3_prob

    
    df.to_csv(output_path, index=False)
    print(f"✅ CSV con probabilidades guardado en: {output_path}")
    print("📊 Columnas: Time(s), Note_21...Note_108, Top1_MIDI, Top1_Prob, Top2_MIDI, ...")


if __name__ == "__main__":
    main()

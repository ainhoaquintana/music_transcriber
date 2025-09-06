import torch
from torch.utils.data import DataLoader, ConcatDataset
from dataset import MaestroDataset, MusicNetDataset
from utils import collate_fn
import os


def main():
    # Paths
    maestro_root = "../data/maestro-v3.0.0/audios"
    musicnet_root = "../data/musicnet_audios"

    # Cargar datasets
    maestro_dataset = MaestroDataset(root_dir=maestro_root)
    musicnet_dataset = MusicNetDataset(root_dir=musicnet_root)

    print(f"Audios Maestro válidos: {len(maestro_dataset)}")
    print(f"Audios MusicNet válidos: {len(musicnet_dataset)}")

    # Unir ambos datasets
    combined_dataset = ConcatDataset([maestro_dataset, musicnet_dataset])
    loader = DataLoader(combined_dataset, batch_size=4, collate_fn=collate_fn)

    onset_sum = 0
    frame_sum = 0
    total_elements = 0

    # Recorremos todos los datos para calcular proporciones
    for mel, onsets, frames in loader:
        onset_sum += onsets.sum().item()
        frame_sum += frames.sum().item()
        total_elements += onsets.numel()

    # Pesos positivos (más grandes si hay mucho desbalance)
    pos_weight_onsets = torch.tensor((total_elements - onset_sum) / onset_sum)
    pos_weight_frames = torch.tensor((total_elements - frame_sum) / frame_sum)

    # Guardar resultados
    os.makedirs("precomputed", exist_ok=True)
    torch.save(pos_weight_onsets, "precomputed/pos_weight_onsets.pt")
    torch.save(pos_weight_frames, "precomputed/pos_weight_frames.pt")

    print("==== Pesos guardados en carpeta precomputed ====")
    print(f"Pos weight onsets: {pos_weight_onsets.item():.4f}")
    print(f"Pos weight frames: {pos_weight_frames.item():.4f}")


if __name__ == "__main__":
    main()

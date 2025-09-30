import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset, random_split
from dataset import MaestroDataset, MusicNetDataset
from utils import collate_fn
from model import CnnTransformerOnsetsFrames


def accuracy(preds, targets, threshold=0.5):
    """Calcula accuracy binario sobre notas activas."""
    preds_bin = (preds > threshold).float()
    correct = (preds_bin == targets).float().sum()
    total = torch.numel(preds_bin)
    return correct / total


def frame_level_f1(preds_bin, targets_bin):
    """Calcula F1-score polifónico a nivel de frame."""
    tp = ((preds_bin == 1) & (targets_bin == 1)).sum().item()
    fp = ((preds_bin == 1) & (targets_bin == 0)).sum().item()
    fn = ((preds_bin == 0) & (targets_bin == 1)).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1, precision, recall


def evaluate_notes(model, dataloader, device, threshold_onset=0.5, threshold_frame=0.5):
    model.eval()
    all_f1 = []
    total_loss = 0.0
    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for mel, onsets_gt, frames_gt in dataloader:
            mel = mel.to(device)
            onsets_gt = onsets_gt.to(device)
            frames_gt = frames_gt.to(device)

            logits_onsets, logits_frames = model(mel)

            # 🔥 DEBUG: activaciones promedio
            mean_onset_sigmoid = torch.sigmoid(logits_onsets).mean().item()
            mean_frame_sigmoid = torch.sigmoid(logits_frames).mean().item()
            print(f"🔥 Mean onset activation: {mean_onset_sigmoid:.6f} | Mean frame activation: {mean_frame_sigmoid:.6f}")

            # calcular loss
            loss_onsets = criterion(logits_onsets, onsets_gt)
            loss_frames = criterion(logits_frames, frames_gt)
            total_loss += (loss_onsets + loss_frames).item() * mel.size(0)

            # binarizar predicciones
            onsets_pred_bin = (torch.sigmoid(logits_onsets) > threshold_onset).int()
            frames_pred_bin = (torch.sigmoid(logits_frames) > threshold_frame).int()

            # calcular F1 por batch
            f1, precision, recall = frame_level_f1(frames_pred_bin, frames_gt.int())
            all_f1.append(f1)

    avg_loss = total_loss / len(dataloader.dataset)
    mean_f1 = np.mean(all_f1)
    return avg_loss, mean_f1


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    maestro_root = "../data/maestro-v3.0.0/audios"
    musicnet_root = "../data/musicnet_audios"

    maestro_dataset = MaestroDataset(maestro_root)
    musicnet_dataset = MusicNetDataset(musicnet_root)

    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.lengths = [len(d) for d in datasets]

        def __len__(self):
            return sum(self.lengths)

        def __getitem__(self, idx):
            for i, l in enumerate(self.lengths):
                if idx < l:
                    return self.datasets[i][idx]
                idx -= l
            raise IndexError

    dataset = CombinedDataset([maestro_dataset, musicnet_dataset])
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    batch_size = 4
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_entrenado_sin_fine_tuning.pth", map_location=device))
    model.to(device)

    # 🔍 Búsqueda de umbrales óptimos
    best_f1 = 0
    best_onset_th, best_frame_th = 0, 0

    for onset_th in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for frame_th in [0.1, 0.15, 0.2, 0.25, 0.3]:
            avg_loss, mean_f1 = evaluate_notes(
                model, test_loader, device,
                threshold_onset=onset_th,
                threshold_frame=frame_th
            )
            print(f"→ Onset {onset_th:.2f}, Frame {frame_th:.2f} → F1: {mean_f1:.4f}")
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_onset_th, best_frame_th = onset_th, frame_th

    print(f"🏆 Mejor F1: {best_f1:.4f} con umbrales Onset={best_onset_th}, Frame={best_frame_th}")


if __name__ == "__main__":
    main()

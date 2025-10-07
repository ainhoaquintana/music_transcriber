# train.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import MaestroDataset, MusicNetDataset, MyDataset
from model import CnnTransformerOnsetsFrames
from utils import collate_fn

import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

@torch.no_grad()
def evaluate_notes(model, dataloader, device, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_frames_preds, all_frames_true = [], []
    all_notes_preds, all_notes_true = [], []

    for mel, onsets, frames in tqdm(dataloader, desc="Evaluating", leave=False):
        mel, onsets, frames = mel.to(device), onsets.to(device), frames.to(device)
        logits_onsets, logits_frames = model(mel)

        # ---- Pérdidas ----
        loss_onsets = F.binary_cross_entropy_with_logits(logits_onsets, onsets)
        loss_frames = F.binary_cross_entropy_with_logits(logits_frames, frames)
        total_loss += (loss_onsets + loss_frames * 5.0).item() * mel.size(0)

        # ---- Predicciones binarizadas ----
        pred_frames = (torch.sigmoid(logits_frames) > threshold).int()
        pred_notes = (torch.sigmoid(logits_onsets) > threshold).int()

        # ---- Guardar todo en CPU para sklearn ----
        all_frames_preds.append(pred_frames.detach().cpu().numpy().flatten())
        all_frames_true.append(frames.detach().cpu().numpy().flatten())
        all_notes_preds.append(pred_notes.detach().cpu().numpy().flatten())
        all_notes_true.append(onsets.detach().cpu().numpy().flatten())

    # Concatenamos todos los lotes
    all_frames_preds = np.concatenate(all_frames_preds)
    all_frames_true = np.concatenate(all_frames_true)
    all_notes_preds = np.concatenate(all_notes_preds)
    all_notes_true = np.concatenate(all_notes_true)

    # ---- Métricas ----
    frame_precision, frame_recall, frame_f1, _ = precision_recall_fscore_support(
        all_frames_true, all_frames_preds, average='binary', zero_division=0
    )
    note_precision, note_recall, note_f1, _ = precision_recall_fscore_support(
        all_notes_true, all_notes_preds, average='binary', zero_division=0
    )

    avg_loss = total_loss / len(dataloader.dataset)

    print("\n📊 === Evaluation Results ===")
    print(f"Loss: {avg_loss:.4f}")
    print(f"[FRAMES] Precision: {frame_precision:.4f} | Recall: {frame_recall:.4f} | F1: {frame_f1:.4f}")
    print(f"[NOTES ] Precision: {note_precision:.4f} | Recall: {note_recall:.4f} | F1: {note_f1:.4f}")

    return avg_loss, note_f1

if __name__ == "__main__":
    BATCH_SIZE = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # === Datasets ===
    musicnet_root = "../data/musicnet_audios"
    maestro_root = "../data/maestro-v3.0.0/audios"
    my_dataset_root = "../data/my_audios"

    maestro_dataset = MaestroDataset(maestro_root)
    musicnet_dataset = MusicNetDataset(musicnet_root)
    my_dataset = MyDataset(my_dataset_root)

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

    dataset = CombinedDataset([musicnet_dataset, maestro_dataset, my_dataset])
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_final_focal.pth", map_location=device))
    model.to(device)
    model.eval()


    avg_loss, mean_f1 = evaluate_notes(model, test_loader, device, threshold=0.5)
    print(f"✅ Test loss: {avg_loss:.4f}, F1 score: {mean_f1:.4f}")

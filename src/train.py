import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import MaestroDataset, MusicNetDataset
from model import CnnTransformerOnsetsFrames
from utils import collate_fn
from test import evaluate

WINDOW_SIZE = 2048
STRIDE = 1024

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def train_one_epoch(model, dataloader, optimizer, criterion_onsets, criterion_frames, device):
    model.train()
    running_loss = 0.0
    for mel, onsets, frames in tqdm(dataloader, desc="Training", leave=False):
        mel = mel.to(device)
        onsets = onsets.to(device)
        frames = frames.to(device)

        optimizer.zero_grad()
        logits_onsets, logits_frames = model(mel)

        loss_onsets = criterion_onsets(logits_onsets, onsets)
        loss_frames = criterion_frames(logits_frames, frames)
        loss = loss_onsets + loss_frames

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * mel.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion_onsets, criterion_frames, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mel, onsets, frames in tqdm(dataloader, desc="Validating", leave=False):
            mel = mel.to(device)
            onsets = onsets.to(device)
            frames = frames.to(device)

            logits_onsets, logits_frames = model(mel)
            loss_onsets = criterion_onsets(logits_onsets, onsets)
            loss_frames = criterion_frames(logits_frames, frames)
            loss = loss_onsets + loss_frames

            val_loss += loss.item() * mel.size(0)
    return val_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Datasets
    maestro_root = "../data/maestro-v3.0.0/audios"
    musicnet_root = "../data/musicnet_audios"

    maestro_dataset = MaestroDataset(maestro_root)
    musicnet_dataset = MusicNetDataset(musicnet_root)

    print(f"Audios Maestro: {len(maestro_dataset)}")
    print(f"Audios MusicNet: {len(musicnet_dataset)}")

    # Combine datasets
    class CombinedDataset(torch.utils.data.Dataset):
        def __init__(self, datasets):
            self.datasets = datasets
            self.lengths = [len(d) for d in datasets]
            print(f"Total combinado: {sum(self.lengths)}")

        def __len__(self):
            return sum(self.lengths)

        def __getitem__(self, idx):
            for i, l in enumerate(self.lengths):
                if idx < l:
                    return self.datasets[i][idx]
                idx -= l
            raise IndexError("Index out of range")

    dataset = CombinedDataset([maestro_dataset, musicnet_dataset])
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    print(f"Train/Val/Test split: {n_train}/{n_val}/{n_test}")

    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    batch_size = 4
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model.to(device)

    # Pos weights
    pos_weight_onsets = torch.load("precomputed/pos_weight_onsets.pt").to(device)
    pos_weight_frames = torch.load("precomputed/pos_weight_frames.pt").to(device)
    criterion_onsets = nn.BCEWithLogitsLoss(pos_weight=pos_weight_onsets)
    criterion_frames = nn.BCEWithLogitsLoss(pos_weight=pos_weight_frames)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    train_losses, val_losses = [], []

    for epoch in range(20):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_onsets, criterion_frames, device)
        val_loss = validate(model, val_loader, criterion_onsets, criterion_frames, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Save best model
    os.makedirs("../checkpoints", exist_ok=True)
    early_stopping.load_best_model(model)
    torch.save(model.state_dict(), "../checkpoints/modelo_entrenado.pth")
    print("Model saved")

    # Learning curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion_onsets, criterion_frames, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

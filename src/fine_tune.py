import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import MaestroDataset, MusicNetDataset
from model import CnnTransformerOnsetsFrames
from utils import collate_fn
from test import evaluate

CHECKPOINT_PATH = "../checkpoints/modelo_entrenado.pth"
FINETUNED_PATH = "../checkpoints/modelo_finetuned.pth"
BATCH_SIZE = 4
LR = 1e-5             # lr más pequeño para fine-tuning
NUM_EPOCHS = 5        # pocas épocas
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    print("Using device:", DEVICE)

    # Load datasets
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
            raise IndexError("Index out of range")

    dataset = CombinedDataset([maestro_dataset, musicnet_dataset])
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Load model checkpoint
    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)  # Las capas nuevas (fc_reduce) se inicializan automáticamente

    model.to(DEVICE)
    model.train()

    # Losses
    pos_weight_onsets = torch.load("precomputed/pos_weight_onsets.pt").to(DEVICE)
    pos_weight_frames = torch.load("precomputed/pos_weight_frames.pt").to(DEVICE)
    criterion_onsets = nn.BCEWithLogitsLoss(pos_weight=pos_weight_onsets)
    criterion_frames = nn.BCEWithLogitsLoss(pos_weight=pos_weight_frames)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # Fine-tuning loop
    train_losses, val_losses = [], []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_onsets, criterion_frames, DEVICE)
        val_loss = validate(model, val_loader, criterion_onsets, criterion_frames, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1} - Fine-tuning Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Save fine-tuned model
    torch.save(model.state_dict(), FINETUNED_PATH)
    print(f"Fine-tuned model saved to {FINETUNED_PATH}")

    # Optional: Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion_onsets, criterion_frames, DEVICE)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # Optional: plot learning curves
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Fine-tuning Learning Curve")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

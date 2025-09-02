import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

from dataset import MaestroDataset
from model import CnnTransformerTranscriber
from test import evaluate
from utils import collate_fn

WINDOW_SIZE = 2048  # frames por ventana
STRIDE = 1024       # overlap entre ventanas

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


def train_one_epoch(model, dataloader, optimizer, criterion_notes, criterion_durs, device, window_size=WINDOW_SIZE, stride=STRIDE):
    model.train()
    running_loss = 0.0
    for mel, notes, durs in tqdm(dataloader, desc="Training", leave=False):
        mel = mel.to(device)
        notes = notes.to(device)
        durs = durs.to(device)

        # Sliding window
        B, T, _ = mel.shape
        start_indices = list(range(0, T, stride))
        optimizer.zero_grad()
        total_loss = 0.0

        for start in start_indices:
            end = min(start + window_size, T)
            mel_chunk = mel[:, start:end, :]
            notes_chunk = notes[:, start:end, :]
            durs_chunk = durs[:, start:end, :]

            logits, dur_preds = model(mel_chunk)
            dur_preds = F.relu(dur_preds)

            loss_notes = criterion_notes(logits, notes_chunk)
            loss_durs = criterion_durs(dur_preds, durs_chunk)
            loss = loss_notes + 0.5 * loss_durs

            loss.backward()
            total_loss += loss.item() * mel_chunk.size(0)

        optimizer.step()
        running_loss += total_loss

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion_notes, criterion_durs, device, window_size=WINDOW_SIZE, stride=STRIDE):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mel, notes, durs in tqdm(dataloader, desc="Validating", leave=False):
            mel = mel.to(device)
            notes = notes.to(device)
            durs = durs.to(device)

            B, T, _ = mel.shape
            start_indices = list(range(0, T, stride))
            total_loss = 0.0

            for start in start_indices:
                end = min(start + window_size, T)
                mel_chunk = mel[:, start:end, :]
                notes_chunk = notes[:, start:end, :]
                durs_chunk = durs[:, start:end, :]

                logits, dur_preds = model(mel_chunk)
                dur_preds = F.relu(dur_preds)
                loss_notes = criterion_notes(logits, notes_chunk)
                loss_durs = criterion_durs(dur_preds, durs_chunk)
                loss = loss_notes + 0.5 * loss_durs
                total_loss += loss.item() * mel_chunk.size(0)

            val_loss += total_loss

    return val_loss / len(dataloader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.cuda.empty_cache()

    # Dataset
    root_dir = "../data/maestro-v3.0.0/audios"
    dataset = MaestroDataset(root_dir=root_dir)

    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    batch_size = 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Modelo
    model = CnnTransformerTranscriber(d_model=512, num_layers=6, nhead=8)
    model.to(device)

    num_epochs = 15
    pos_weight = torch.load("precomputed/pos_weight.pt").to(device)
    criterion_notes = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_durs = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_notes, criterion_durs, device)
        val_loss = validate(model, val_loader, criterion_notes, criterion_durs, device)
        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        # Diagnostics
        mel_batch, notes_batch, _ = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            mel_batch = mel_batch.to(device)
            logits, _ = model(mel_batch)
            probs = torch.sigmoid(logits)
            avg_probs = probs.mean(dim=(0,1)).cpu().numpy()
            plt.bar(range(88), avg_probs)
            plt.xlabel("MIDI note (0–87)")
            plt.ylabel("Average predicted probability")
            plt.title(f"Epoch {epoch+1} note activations")
            os.makedirs("plots/diagnostics", exist_ok=True)
            plt.savefig(f"plots/diagnostics/epoch_{epoch+1}_note_probs.png")
            plt.close()
        model.train()
        torch.cuda.empty_cache()

    # Guardar mejor modelo
    model_path = "../checkpoints/modelo_entrenado.pth"
    os.makedirs("../checkpoints", exist_ok=True)
    early_stopping.load_best_model(model)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Learning curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # Test
    test_loss, test_acc = evaluate(model, test_loader, criterion_notes, criterion_durs, device)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

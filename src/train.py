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
from test import evaluate_notes

# ===== HYPERPARAMS =====
EPOCHS = 40
BATCH_SIZE = 4
LR = 1e-5
WEIGHT_DECAY = 1e-4
FRAME_LOSS_FACTOR = 5.0
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 6


# ===== FOCAL LOSS =====
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal = self.alpha * (1 - pt) ** self.gamma * bce

        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        else:
            return focal


# ===== EARLY STOPPING =====
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


# ===== TRAINING =====
def train_one_epoch(model, dataloader, optimizer, criterion_onsets, criterion_frames, device, epoch):
    model.train()
    running_loss = 0.0
    total_batches = 0
    onset_act, frame_act = 0.0, 0.0

    for mel, onsets, frames in tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=False):
        mel, onsets, frames = mel.to(device), onsets.to(device), frames.to(device)

        optimizer.zero_grad()
        logits_onsets, logits_frames = model(mel)

        loss_onsets = criterion_onsets(logits_onsets, onsets)
        loss_frames = criterion_frames(logits_frames, frames)
        loss = loss_onsets + FRAME_LOSS_FACTOR * loss_frames

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        running_loss += loss.item() * mel.size(0)
        onset_act += torch.sigmoid(logits_onsets).mean().item()
        frame_act += torch.sigmoid(logits_frames).mean().item()
        total_batches += 1

    avg_loss = running_loss / len(dataloader.dataset)
    print(f"Train Loss: {avg_loss:.4f} | Onset Act: {onset_act / total_batches:.4f} | Frame Act: {frame_act / total_batches:.4f}")
    return avg_loss


# ===== VALIDATION =====
def validate(model, dataloader, criterion_onsets, criterion_frames, device, epoch):
    model.eval()
    val_loss = 0.0
    total_batches = 0
    onset_act, frame_act = 0.0, 0.0

    with torch.no_grad():
        for mel, onsets, frames in tqdm(dataloader, desc=f"Validating Epoch {epoch+1}", leave=False):
            mel, onsets, frames = mel.to(device), onsets.to(device), frames.to(device)
            logits_onsets, logits_frames = model(mel)

            loss_onsets = criterion_onsets(logits_onsets, onsets)
            loss_frames = criterion_frames(logits_frames, frames)
            loss = loss_onsets + FRAME_LOSS_FACTOR * loss_frames

            val_loss += loss.item() * mel.size(0)
            onset_act += torch.sigmoid(logits_onsets).mean().item()
            frame_act += torch.sigmoid(logits_frames).mean().item()
            total_batches += 1

    avg_val_loss = val_loss / len(dataloader.dataset)
    print(f"Val Loss: {avg_val_loss:.4f} | Onset Act: {onset_act / total_batches:.4f} | Frame Act: {frame_act / total_batches:.4f}")
    return avg_val_loss


# ===== MAIN =====
def main():
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

    # === Modelo ===
    model = CnnTransformerOnsetsFrames(n_mels=229, d_model=512, num_layers=6, nhead=8).to(device)

    # === Pérdidas ===
    criterion_onsets = FocalLoss(alpha=0.75, gamma=2.0)
    criterion_frames = FocalLoss(alpha=1.25, gamma=3.0)

    # === Optimizador & Scheduler ===
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, delta=0.01)

    train_losses, val_losses = [], []

    # === Training Loop ===
    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_onsets, criterion_frames, device, epoch)
        val_loss = validate(model, val_loader, criterion_onsets, criterion_frames, device, epoch)

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("⚠️ Early stopping triggered.")
            break

    # === Save Best Model ===
    os.makedirs("../checkpoints", exist_ok=True)
    early_stopping.load_best_model(model)
    torch.save(model.state_dict(), "../checkpoints/modelo_final_focal.pth")
    print("✅ Best model saved!")

    # === Plot Learning Curve ===
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()

    # === Final Test ===
    avg_loss, mean_f1 = evaluate_notes(model, test_loader, device, threshold=0.5)
    print(f"✅ Test loss: {avg_loss:.4f}, F1 score: {mean_f1:.4f}")



if __name__ == "__main__":
    main()

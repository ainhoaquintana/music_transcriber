import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import MyDataset 
from model import CnnTransformerOnsetsFrames
from utils import collate_fn

BATCH_SIZE = 4
LR = 1e-5
WEIGHT_DECAY = 1e-4
EPOCHS = 20
FRAME_LOSS_FACTOR = 5.0
MAX_GRAD_NORM = 1.0
EARLY_STOP_PATIENCE = 3

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss.sum()

class EarlyStopping:
    def __init__(self, patience=3, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss + self.delta < self.best_loss:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_best_model(self, model):
        model.load_state_dict(self.best_state)

def train_one_epoch(model, loader, optimizer, criterion_onsets, criterion_frames, device):
    model.train()
    running_loss = 0.0
    for mel, onsets, frames in tqdm(loader, desc="Training", leave=False):
        mel, onsets, frames = mel.to(device), onsets.to(device), frames.to(device)
        optimizer.zero_grad()
        logits_onsets, logits_frames = model(mel)
        loss_onsets = criterion_onsets(logits_onsets, onsets)
        loss_frames = criterion_frames(logits_frames, frames)
        loss = 0.5 * loss_onsets + FRAME_LOSS_FACTOR * loss_frames
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        running_loss += loss.item() * mel.size(0)
    return running_loss / len(loader.dataset)

def validate(model, loader, criterion_onsets, criterion_frames, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mel, onsets, frames in tqdm(loader, desc="Validating", leave=False):
            mel, onsets, frames = mel.to(device), onsets.to(device), frames.to(device)
            logits_onsets, logits_frames = model(mel)
            loss_onsets = criterion_onsets(logits_onsets, onsets)
            loss_frames = criterion_frames(logits_frames, frames)
            val_loss += (0.5 * loss_onsets + FRAME_LOSS_FACTOR * loss_frames).item() * mel.size(0)
    return val_loss / len(loader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset_root = "../data/my_audios"
    dataset = MyDataset(dataset_root)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model.load_state_dict(torch.load("../checkpoints/modelo_final_focal.pth", map_location=device))
    model.to(device)

    criterion_onsets = FocalLoss(alpha=0.75, gamma=2.0)
    criterion_frames = FocalLoss(alpha=1.5, gamma=3.0)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    for epoch in range(EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{EPOCHS} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_onsets, criterion_frames, device)
        val_loss = validate(model, val_loader, criterion_onsets, criterion_frames, device)
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Guardamos el mejor modelo
    os.makedirs("../checkpoints", exist_ok=True)
    early_stopping.load_best_model(model)
    torch.save(model.state_dict(), "../checkpoints/modelo_finetuned_monophonic.pth")
    print("Fine-tuned monophonic model saved!")

if __name__ == "__main__":
    main()

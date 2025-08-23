import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from dataset import MaestroDataset
from model import CnnTransformerTranscriber
import numpy as np
from test import evaluate
from utils import collate_fn
from tqdm import tqdm

MAX_LEN = 2048  

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

def train_one_epoch(model, dataloader, optimizer, criterion_notes, criterion_durs, device):
    model.train()
    running_loss = 0.0
    for mel, notes, durs in tqdm(dataloader, desc="Training", leave=False):
        mel = mel.to(device)          # (batch, T, n_mels)
        notes = notes.to(device)      # (batch, T, 88)
        durs = durs.to(device)        # (batch, T, 88)
        
        # if device.type == "cuda":
        #     print("After data to device:")
        #     print(torch.cuda.memory_summary())

        
        optimizer.zero_grad()
        logits, dur_preds = model(mel) # logits: (B, T_reduced, n_notes), dur_preds: (B, T_reduced, n_notes)

        loss_notes = criterion_notes(logits, notes)
        loss_durs = criterion_durs(dur_preds, durs)
        # loss = loss_notes + loss_durs
        loss = loss_notes + 0.5 * loss_durs ##################################


        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel.size(0)

    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion_notes, criterion_durs, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for mel, notes, durs in tqdm(dataloader, desc="Validating", leave=False):
            mel = mel.to(device)          # (batch, T, n_mels)
            notes = notes.to(device)      # (batch, T, 88)
            durs = durs.to(device)        # (batch, T, 88)
            logits, dur_preds = model(mel)

            loss_notes = criterion_notes(logits, notes)
            loss_durs = criterion_durs(dur_preds, durs)
            # loss = loss_notes + loss_durs
            loss = loss_notes + 0.5 * loss_durs ##################################


            val_loss += loss.item() * mel.size(0)

    return val_loss / len(dataloader.dataset)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    torch.cuda.empty_cache() 

    # Cargar dataset
    root_dir = "../data/maestro-v3.0.0/audios"
    # root_dir = "../data/maestro-v3.0.0"  #
    dataset = MaestroDataset(root_dir=root_dir)

    # Split into train (70%), val (15%), test (15%)
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    batch_size = 2
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = CnnTransformerTranscriber(d_model=512, num_layers=6, nhead=8) 
    model.to(device)

    num_epochs = 15
    
    criterion_notes = nn.BCEWithLogitsLoss()
    criterion_durs = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)#########################
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # early_stopping = EarlyStopping(patience=5, delta=0.01)

    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_notes, criterion_durs, device)
        val_loss = validate(model, val_loader, criterion_notes, criterion_durs, device)

        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.4f} - Val loss: {val_loss:.4f} -  LR: {scheduler.get_last_lr()[0]:.6f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

         # Visualize mel spectrogram and note targets for the first batch
        mel_batch, notes_batch, durs_batch = next(iter(train_loader))
        mel = mel_batch[0].cpu().numpy().T  # (n_mels, T)
        notes = notes_batch[0].cpu().numpy().T  # (88, T)

        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
        plt.title('Mel Spectrogram (Training Sample)')
        plt.ylabel('Mel bins')
        plt.colorbar()

        plt.subplot(2, 1, 2)
        plt.imshow(notes, aspect='auto', origin='lower', cmap='hot')
        plt.title('Note Targets (Training Sample)')
        plt.xlabel('Time (frames)')
        plt.ylabel('MIDI pitch')
        plt.colorbar()

        plt.tight_layout()
        # plt.show()
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig(f"plots/epoch_{epoch+1}.png")
        plt.close()

    #     early_stopping(val_loss, model)
    #     if early_stopping.early_stop:
    #         print("Early stopped")
    #         break

    # early_stopping.load_best_model(model)
      
        # if device.type == "cuda":
        #     print(torch.cuda.memory_summary())

    # Guardamos el modelo entrenado
    model_path = "../checkpoints/modelo_entrenado.pth"
    if not os.path.exists("../checkpoints"):
        os.makedirs("../checkpoints")
    print(f"Saving model to {model_path}")
    torch.save(model.state_dict(), model_path)

    # Visualize learning curve
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.show()
    
    # Evaluate on test set
    test_loss, test_acc = evaluate(model, test_loader, criterion_notes, criterion_durs, device)
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()

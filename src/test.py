import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import MaestroDataset
from model import CnnTransformerTranscriber
from utils import collate_fn
import torch

def accuracy(preds, targets, threshold=0.5):
    # preds and targets shape: (N, T, n_notes)
    preds_bin = (preds > threshold).float()
    correct = (preds_bin == targets).float().sum()
    total = torch.numel(preds_bin)
    return correct / total

def evaluate(model, dataloader, criterion_notes, criterion_durs, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for mel, notes, durs in dataloader:
            mel = mel.to(device)
            notes = notes.to(device)
            durs = durs.to(device)
            logits, dur_preds = model(mel)
            loss_notes = criterion_notes(logits, notes)
            loss_durs = criterion_durs(dur_preds, durs)
            loss = loss_notes + loss_durs
            test_loss += loss.item() * mel.size(0)
            all_preds.append(torch.sigmoid(logits).cpu())
            all_targets.append(notes.cpu())
    avg_loss = test_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    acc = accuracy(preds, targets)
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CnnTransformerTranscriber()
    model_path = "../checkpoints/modelo_entrenado.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Prepare your test dataset and loader
    root_dir = "../data/maestro-v3.0.0/2008"
    dataset = MaestroDataset(root_dir=root_dir)
    # Split or select your test set as needed
    batch_size = 4
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    criterion_notes = nn.BCEWithLogitsLoss()
    criterion_durs = nn.MSELoss()

    test_loss, test_acc = evaluate(model, test_loader, criterion_notes, criterion_durs, device)
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()

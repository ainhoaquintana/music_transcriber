import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from dataset import MaestroDataset, MusicNetDataset
from utils import collate_fn
from model import CnnTransformerOnsetsFrames


def accuracy(preds, targets, threshold=0.5):
    """
    Calcula accuracy binario sobre notas activas.
    preds, targets: tensores (N, T, 88)
    """
    preds_bin = (preds > threshold).float()
    correct = (preds_bin == targets).float().sum()
    total = torch.numel(preds_bin)
    return correct / total


def evaluate(model, dataloader, criterion_onsets, criterion_frames, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for mel, onsets, frames in dataloader:
            mel = mel.to(device)
            onsets = onsets.to(device)
            frames = frames.to(device)

            logits_onsets, logits_frames = model(mel)

            loss_onsets = criterion_onsets(logits_onsets, onsets)
            loss_frames = criterion_frames(logits_frames, frames)
            loss = loss_onsets + loss_frames
            total_loss += loss.item() * mel.size(0)

            all_preds.append(torch.sigmoid(logits_onsets).cpu())
            all_targets.append(onsets.cpu())

    avg_loss = total_loss / len(dataloader.dataset)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    acc = accuracy(preds, targets)
    return avg_loss, acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets combinados (test directo sobre todo MusicNet + Maestro)
    maestro = MaestroDataset(root_dir="../data/maestro-v3.0.0/audios")
    musicnet = MusicNetDataset(root_dir="../data/musicnet_audios")
    dataset = ConcatDataset([maestro, musicnet])

    test_loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

    # Modelo
    model = CnnTransformerOnsetsFrames(d_model=512, num_layers=6, nhead=8)
    model_path = "../checkpoints/modelo_entrenado.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Cargar pesos precomputados
    pos_weight_onsets = torch.load("precomputed/pos_weight_onsets.pt").to(device)
    pos_weight_frames = torch.load("precomputed/pos_weight_frames.pt").to(device)
    criterion_onsets = nn.BCEWithLogitsLoss(pos_weight=pos_weight_onsets)
    criterion_frames = nn.BCEWithLogitsLoss(pos_weight=pos_weight_frames)

    # Evaluar
    test_loss, test_acc = evaluate(model, test_loader, criterion_onsets, criterion_frames, device)
    print(f"Test loss: {test_loss:.4f} - Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

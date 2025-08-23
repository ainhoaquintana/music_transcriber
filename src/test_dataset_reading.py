from dataset import MaestroDataset
from model import CnnTransformerTranscriber
import torch

dataset = MaestroDataset("..\data\maestro-v3.0.0")
# print("Samples:", len(dataset))
mel, notes, durs = dataset[0]
print("mel:", mel.shape, "notes:", notes.shape, "durs:", durs.shape)

model = CnnTransformerTranscriber()
model.eval()
with torch.no_grad():
    logits, dur_preds = model(mel.unsqueeze(0))
    print("logits:", logits.shape, "dur_preds:", dur_preds.shape)



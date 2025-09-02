import os
import torch
from itertools import islice
from torch.utils.data import DataLoader
from dataset import MaestroDataset
from utils import collate_fn

# -------------------- CONFIG --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
root_dir = "../data/maestro-v3.0.0/audios"
num_batches = 200  # number of batches to estimate pos_weight
batch_size = 2
save_path = "precomputed/pos_weight.pt"
# ------------------------------------------------

# Create dataset and DataLoader
dataset = MaestroDataset(root_dir=root_dir)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Initialize counters
total_pos = torch.zeros(88, dtype=torch.long)
total_neg = torch.zeros(88, dtype=torch.long)

print(f"Computing pos_weight using the first {num_batches} batches...")

# Iterate only over the first num_batches batches
for i, (_, notes, _) in enumerate(islice(loader, num_batches)):
    notes = notes.view(-1, 88)  # flatten (B*T, 88)
    total_pos += notes.sum(dim=0).long()
    total_neg += (1 - notes).sum(dim=0).long()

# Compute pos_weight
pos_weight = total_neg.float() / (total_pos.float() + 1e-6)
pos_weight = pos_weight.to(device)

# Save to file
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(pos_weight.cpu(), save_path)

print(f"pos_weight saved to {save_path}")
print("Sample pos_weight (first 10 notes):", pos_weight[:10].cpu().numpy())
print("Computation completed.")
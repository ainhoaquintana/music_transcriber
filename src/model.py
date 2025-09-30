import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class CnnTransformerOnsetsFrames(nn.Module):
    def __init__(self, n_mels=229, n_notes=88, d_model=512, nhead=8, num_layers=6):
        super().__init__()

        # CNN feature extractor 
        self.cnn = nn.Sequential(
            # Bloque 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout(0.2),  # regularización leve

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout(0.2),

            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,1)),
            nn.Dropout(0.3),  # un poco más fuerte aquí para prevenir overfitting

            # Bloque 4
            nn.Conv2d(128, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Dropout(0.3)  # último bloque antes del transformer
        )

        # Positional encoding y Transformer
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=0.2
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.fc_onsets = nn.Linear(d_model, n_notes)
        self.fc_frames = nn.Linear(d_model + n_notes, n_notes)  # concatenamos onsets

    def forward(self, mel):
        """
        mel: (B, T, n_mels)
        Retorna:
            onsets_logits: (B, T, n_notes)
            frames_logits: (B, T, n_notes)
        """
        x = mel.permute(0, 2, 1).unsqueeze(1)  # (B, 1, n_mels, T)
        x = self.cnn(x)                        # (B, d_model, n_mels', T_reduced)
        x = x.mean(2)                          # promedio sobre frecuencia -> (B, d_model, T)
        x = x.permute(0, 2, 1)                 # (B, T, d_model)

        # Positional encoding + Transformer
        x = self.pos_encoder(x)
        x = self.transformer(x)

        # Cabezal Onsets
        onsets_logits = self.fc_onsets(x)  # (B, T, n_notes)

        if onsets_logits.size(-1) > self.fc_frames.in_features - x.size(-1):
            onsets_logits = onsets_logits[:, :, :self.fc_frames.in_features - x.size(-1)]
        
        x_combined = torch.cat([x, onsets_logits.detach()], dim=-1)
        frames_logits = self.fc_frames(x_combined)  # (B, T, n_notes)

        return onsets_logits, frames_logits

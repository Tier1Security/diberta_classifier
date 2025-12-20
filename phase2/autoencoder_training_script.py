"""Autoencoder training script converted from the notebook `Autoencoder_Training_Script.ipynb`.

Usage:
    python3 phase2/autoencoder_training_script.py --csv PATH_TO_CSV --save-model anomaly_engine.pt
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm


class RobertaAutoencoder(nn.Module):
    def __init__(self, model_name: str = "roberta-base"):
        super(RobertaAutoencoder, self).__init__()
        # Load encoder (RoBERTa)
        self.encoder = RobertaModel.from_pretrained(model_name)

        # Decoder: reconstruct the 768-dim RoBERTa CLS embedding
        self.decoder = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 768)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :]
        reconstructed = self.decoder(embedding)
        return embedding, reconstructed


class BaselineDataset(Dataset):
    def __init__(self, csv_file: str, tokenizer: RobertaTokenizer, max_len: int = 128):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["command"])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def train_anomaly_engine(
    csv_path: str,
    model_save_path: str = "anomaly_engine.pt",
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 1e-5,
    max_len: int = 128,
    device: str | None = None,
):
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaAutoencoder().to(device)

    dataset = BaselineDataset(csv_path, tokenizer, max_len=max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"[*] Training Anomaly Engine on {len(dataset)} benign samples...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)

            emb, rec = model(input_ids, mask)
            loss = criterion(rec, emb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader) if len(loader) > 0 else float("nan")
        print(f"Epoch {epoch+1} Loss: {avg_loss:.6f}")

    # Compute anomaly threshold using reconstruction errors
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emb, rec = model(input_ids, mask)
            error = torch.mean((emb - rec) ** 2, dim=1)
            errors.extend(error.cpu().numpy())

    threshold = float(np.mean(errors) + 3 * np.std(errors))
    print(f"[+] Training Complete. Anomaly Threshold set to: {threshold}")

    state = {
        "model_state": model.state_dict(),
        "threshold": threshold,
    }
    torch.save(state, model_save_path)
    print(f"[+] Saved model and threshold to: {model_save_path}")
    return model_save_path, threshold


def _parse_args():
    p = argparse.ArgumentParser(description="Train RoBERTa autoencoder for anomaly detection")
    p.add_argument("--csv", required=True, help="Path to benign baseline CSV")
    p.add_argument("--save-model", default="anomaly_engine.pt", help="Path to save trained model and threshold")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--device", type=str, default=None, help="Torch device string, e.g. 'cpu' or 'cuda:0' (optional)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_anomaly_engine(
        csv_path=args.csv,
        model_save_path=args.save_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_len=args.max_len,
        device=args.device,
    )

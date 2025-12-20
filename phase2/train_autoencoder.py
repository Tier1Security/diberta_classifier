import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# ==========================================
# 1. Synchronized Architecture
# ==========================================
class SecurityAgentX_Model(nn.Module):
    """
    Heuristic Anomaly Architecture: Shared RoBERTa Encoder with 
    an Autoencoder Decoder for reconstruction-based detection.
    Matches the architecture in inference_engine.py exactly.
    """
    def __init__(self, model_name: str = "roberta-base"):
        super(SecurityAgentX_Model, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        
        # Phase 2 Head: Autoencoder Decoder
        self.decoder_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(), # Optimized for ADA architecture
            nn.Dropout(0.1),
            nn.Linear(1024, 768)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :] # CLS token (The Latent Vector)
        
        reconstructed = self.decoder_head(embedding)
        return embedding, reconstructed

# ==========================================
# 2. Optimized Dataset Handler
# ==========================================
class BaselineDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_len=128):
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

# ==========================================
# 3. Training Loop (RTX 4000 ADA Optimized)
# ==========================================
def train_pure_heuristic(
    csv_path, 
    model_save_path="anomaly_engine.pt", 
    epochs=3, 
    batch_size=128,
    lr=2e-5
):
    # Setup Device & Performance Flags
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True # TF32 for ADA
        torch.backends.cudnn.benchmark = True
        print(f"[*] Optimized for: {torch.cuda.get_device_name(0)}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = SecurityAgentX_Model().to(device)
    
    # RTX 4000 ADA Optimization: Kernel Fusion
    try:
        print("[*] Compiling model for ADA architecture...")
        model = torch.compile(model)
    except Exception as e:
        print(f"[!] Compilation skipped: {e}")

    dataset = BaselineDataset(csv_path, tokenizer)
    # High-throughput loading for i7-13700K
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True,
        prefetch_factor=2
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    # Use bfloat16 for RTX 4000 ADA
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[*] Starting Heuristic Training on {len(dataset)} benign samples...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in loop:
            optimizer.zero_grad(set_to_none=True)
            
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)

            with autocast(dtype=dtype):
                emb, rec = model(ids, mask)
                loss = criterion(rec, emb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.8f}")

    # ==========================================
    # 4. Threshold Calculation (Z-Score)
    # ==========================================
    print("[*] Training Complete. Calculating Anomaly Threshold...")
    model.eval()
    errors = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Baseline MSE Scan"):
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            
            with autocast(dtype=dtype):
                emb, rec = model(ids, mask)
                # Row-wise MSE
                error = torch.mean((emb - rec) ** 2, dim=1)
            
            errors.extend(error.float().cpu().numpy())

    mean_err = np.mean(errors)
    std_err = np.std(errors)
    threshold = float(mean_err + (3 * std_err))
    
    print(f"\n[+] Manifold Established.")
    print(f"[+] Mean Baseline Error: {mean_err:.8f}")
    print(f"[+] Anomaly Threshold (Z=3): {threshold:.8f}")

    # Extract original model if compiled
    final_state = model._orig_mod if hasattr(model, '_orig_mod') else model
    
    torch.save({
        "model_state": final_state.state_dict(),
        "threshold": threshold,
        "arch": "heuristic-autoencoder-v2"
    }, model_save_path)
    
    print(f"[+] Saved Hybrid Brain to: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pure Heuristic Training for RTX 4000 ADA")
    parser.add_argument("--csv", required=True, help="Path to synthetic_benign_baseline.csv")
    parser.add_argument("--save-model", default="anomaly_engine.pt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    
    train_pure_heuristic(
        csv_path=args.csv,
        model_save_path=args.save_model,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
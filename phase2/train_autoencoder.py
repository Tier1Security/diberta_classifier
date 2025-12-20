import argparse
import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import RobertaModel, RobertaTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ==========================================
# 1. Hardware-Specific Optimization
# ==========================================
torch.set_num_threads(8) 
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Use expandable segments to help with fragmentation
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

def clear_gpu_memory():
    """Flushes cache and garbage collects to prevent OOM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

class SecurityAgentX_Model(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        super(SecurityAgentX_Model, self).__init__()
        # CodeBERT provides superior syntactic understanding of command-line operators
        self.encoder = RobertaModel.from_pretrained(model_name)
        
        # MEMORY SAFETY: Enable gradient checkpointing for adversarial passes
        self.encoder.gradient_checkpointing_enable()
        
        # Deep bottleneck decoder for reconstruction-based anomaly detection
        self.decoder_head = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 768)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embedding = outputs.last_hidden_state[:, 0, :]
        reconstructed = self.decoder_head(embedding)
        return embedding, reconstructed

class AdversarialDataset(Dataset):
    """
    Feeds both Benign and Malicious pairs to the model to force manifold divergence.
    """
    def __init__(self, benign_ids, benign_mask, mal_ids, mal_mask):
        self.benign_ids = benign_ids
        self.benign_mask = benign_mask
        self.mal_ids = mal_ids
        self.mal_mask = mal_mask
        self.mal_len = len(mal_ids)

    def __len__(self):
        return len(self.benign_ids)

    def __getitem__(self, idx):
        # Oversample malicious data to match benign dataset size for balanced loss pressure
        mal_idx = idx % self.mal_len
        return {
            "b_ids": self.benign_ids[idx],
            "b_mask": self.benign_mask[idx],
            "m_ids": self.mal_ids[mal_idx],
            "m_mask": self.mal_mask[mal_idx]
        }

def train_adversarial_engine(
    benign_csv, 
    mal_csv,
    model_save_path="anomaly_engine.pt", 
    total_epochs=5, 
    batch_size=128, 
    margin=0.015,
    eval_steps=100 # Checkpointing every 100 steps
):
    clear_gpu_memory()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    # --- LOAD AND PRE-TOKENIZE ---
    print(f"[*] Pre-tokenizing Benign and Malicious datasets...")
    b_df = pd.read_csv(benign_csv)
    m_df = pd.read_csv(mal_csv)
    
    b_enc = tokenizer(b_df['command'].astype(str).tolist(), padding="max_length", truncation=True, max_length=96, return_tensors="pt")
    m_enc = tokenizer(m_df['command'].astype(str).tolist(), padding="max_length", truncation=True, max_length=96, return_tensors="pt")
    
    full_dataset = AdversarialDataset(b_enc['input_ids'], b_enc['attention_mask'], m_enc['input_ids'], m_enc['attention_mask'])
    
    # Split into train and validation for honest checkpointing
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SecurityAgentX_Model().to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5},
        {'params': model.decoder_head.parameters(), 'lr': 1e-4}
    ], weight_decay=0.01)

    total_steps = len(train_loader) * total_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*total_steps), num_training_steps=total_steps)
    
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')

    # --- RESUME LOGIC ---
    if os.path.exists(model_save_path):
        print(f"[*] Found existing checkpoint {model_save_path}. Loading...")
        checkpoint = torch.load(model_save_path, map_location=device)
        state_dict = checkpoint.get('model_state', checkpoint)
        # Clean torch.compile prefixes if they exist
        clean_state = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(clean_state, strict=False)
        
        if isinstance(checkpoint, dict):
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            if 'optimizer_state' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            print(f"[+] Resumed from Epoch {start_epoch}, Step {global_step}")

    criterion = nn.MSELoss()
    scaler = GradScaler()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    print(f"[*] Starting Adversarial Training (CodeBERT Base). Target SNR: > 10.0x")

    model.train()
    for epoch in range(start_epoch, total_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}")
        for batch in loop:
            optimizer.zero_grad(set_to_none=True)
            
            b_ids, b_mask = batch["b_ids"].to(device), batch["b_mask"].to(device)
            m_ids, m_mask = batch["m_ids"].to(device), batch["m_mask"].to(device)

            with autocast('cuda', dtype=dtype):
                # 1. Benign Forward Pass
                b_emb, b_rec = model(b_ids, b_mask)
                b_loss = criterion(b_rec, b_emb)
                
                # 2. Malicious Forward Pass
                m_emb, m_rec = model(m_ids, m_mask)
                m_mse = torch.mean((m_emb - m_rec)**2, dim=1)
                
                # Contrastive Penalty
                m_loss = torch.mean(torch.clamp(margin - m_mse, min=0.0))
                
                total_loss = b_loss + m_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            global_step += 1
            rate = loop.format_dict.get('rate')
            it_s_str = f"{rate:.1f}" if rate is not None else "0.0"
            loop.set_postfix(b_loss=f"{b_loss.item():.6f}", m_loss=f"{m_loss.item():.6f}", it_s=it_s_str)

            # --- PERIODIC CHECKPOINTING & VALIDATION ---
            if global_step % eval_steps == 0:
                model.eval()
                v_loss = 0
                with torch.no_grad():
                    for v_batch in val_loader:
                        vb_ids = v_batch["b_ids"].to(device)
                        vb_mask = v_batch["b_mask"].to(device)
                        with autocast('cuda', dtype=dtype):
                            ve, vr = model(vb_ids, vb_mask)
                            v_loss += criterion(vr, ve).item()
                
                avg_v_loss = v_loss / len(val_loader)
                if avg_v_loss < best_val_loss:
                    best_val_loss = avg_v_loss
                    print(f"\n[Step {global_step}] New Best Val Loss: {avg_v_loss:.8f}. Saving checkpoint...")
                    torch.save({
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "val_loss": avg_v_loss
                    }, model_save_path)
                model.train()

    # Final Thresholding
    print("[*] Calculating final production threshold...")
    model.eval()
    clear_gpu_memory()
    b_errors = []
    with torch.no_grad():
        for batch in train_loader:
            ids, mask = batch["b_ids"].to(device), batch["b_mask"].to(device)
            with autocast('cuda', dtype=dtype):
                e, r = model(ids, mask)
                b_errors.extend(torch.mean((e-r)**2, dim=1).float().cpu().numpy())
    
    threshold = float(np.mean(b_errors) + 3 * np.std(b_errors))
    
    # Save final production model with threshold
    final_checkpoint = torch.load(model_save_path) if os.path.exists(model_save_path) else {}
    final_checkpoint.update({"model_state": model.state_dict(), "threshold": threshold})
    torch.save(final_checkpoint, model_save_path)
    print(f"[+] CodeBERT Adversarial Brain saved as: {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benign", default="synthetic_benign_baseline.csv")
    parser.add_argument("--malicious", default="mitre_atlas_raw.csv")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()
    
    train_adversarial_engine(benign_csv=args.benign, mal_csv=args.malicious, total_epochs=args.epochs)
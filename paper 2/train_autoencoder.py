import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.amp import autocast, GradScaler

# ==========================================
# 1. SHARED NORMALIZATION LOGIC
# ==========================================
def normalize_command(cmd):
    """
    The 'Sanitizer' - This is the only version of the command the model sees.
    """
    if not isinstance(cmd, str): return ""
    cmd = cmd.lower()
    cmd = cmd.replace("^", "").replace("`", "")
    cmd = cmd.replace("/", "\\")
    cmd = " ".join(cmd.split())
    cmd = cmd.replace("> nul 2>&1", "").replace(">nul", "")
    cmd = cmd.replace("\\.\\", "\\")
    return cmd.strip()

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class SecurityAgentX_Model(nn.Module):
    def __init__(self, model_name: str = "microsoft/codebert-base"):
        super(SecurityAgentX_Model, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        # Deeper head to learn the complex nuances of normalized skeletons
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

class ContrastiveNormalizedDataset(Dataset):
    def __init__(self, benign_cmds, mal_cmds, tokenizer):
        # We ensure everything is normalized during dataset initialization
        self.benign = [normalize_command(c) for c in benign_cmds]
        self.mal = [normalize_command(c) for c in mal_cmds]
        self.tokenizer = tokenizer
        self.mal_len = len(self.mal)

    def __len__(self):
        return len(self.benign)

    def __getitem__(self, idx):
        return {
            "benign": self.benign[idx],
            "malicious": self.mal[idx % self.mal_len]
        }

# ==========================================
# 3. CONTRASTIVE TRAINING LOOP
# ==========================================
def train_hardened_engine(benign_csv, mal_csv, epochs=3, margin=0.008):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    
    b_df = pd.read_csv(benign_csv)
    m_df = pd.read_csv(mal_csv)
    
    dataset = ContrastiveNormalizedDataset(b_df['command'].tolist(), m_df['command'].tolist(), tokenizer)
    
    def collate_fn(batch):
        def tokenize(cmds):
            return tokenizer(cmds, padding="max_length", truncation=True, max_length=96, return_tensors="pt")
        return {
            "benign": tokenize([x["benign"] for x in batch]),
            "malicious": tokenize([x["malicious"] for x in batch])
        }

    loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    model = SecurityAgentX_Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-5)
    scaler = GradScaler()
    mse_loss = nn.MSELoss()
    
    print(f"[*] Training on NORMALIZED data only. Target Safety Margin: {margin}")

    model.train()
    for epoch in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            optimizer.zero_grad()
            
            b_in = {k: v.to(device) for k, v in batch["benign"].items()}
            m_in = {k: v.to(device) for k, v in batch["malicious"].items()}

            with autocast('cuda'):
                # 1. Benign Objective: MINIMIZE error for normalized safe skeletons
                e_b, r_b = model(b_in["input_ids"], b_in["attention_mask"])
                loss_benign = mse_loss(r_b, e_b)
                
                # 2. Malicious Objective: MAXIMIZE error for normalized evil skeletons
                e_m, r_m = model(m_in["input_ids"], m_in["attention_mask"])
                mse_malicious = torch.mean((e_m - r_m)**2, dim=1)
                
                # Force malicious error to be at least 'margin' above benign
                loss_malicious = torch.mean(torch.clamp(margin - mse_malicious, min=0.0))
                
                # We weight the malicious penalty heavily to ensure separation
                total_loss = loss_benign + (loss_malicious * 3.0)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            loop.set_postfix(b_err=f"{loss_benign.item():.6f}", m_pen=f"{loss_malicious.item():.6f}")

    # Final threshold calculation based on normalized baseline
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            b_in = {k: v.to(device) for k, v in batch["benign"].items()}
            e, r = model(b_in["input_ids"], b_in["attention_mask"])
            errors.extend(torch.mean((e-r)**2, dim=1).cpu().numpy())
    
    threshold = float(np.mean(errors) + 4 * np.std(errors))
    torch.save({"model_state": model.state_dict(), "threshold": threshold}, "anomaly_engine.pt")
    print(f"[+] Normalized Brain Saved. Threshold: {threshold:.8f}")

if __name__ == "__main__":
    train_hardened_engine("synthetic_benign_baseline.csv", "mitre_atlas_raw.csv")
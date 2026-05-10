import torch
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
from tqdm import tqdm
import os

try:
    from train_autoencoder import SecurityAgentX_Model, normalize_command
except ImportError:
    print("[!] Error: train_autoencoder.py must be in current directory.")
    exit()

class NormalizedEvaluator:
    def __init__(self, model_path="anomaly_engine.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SecurityAgentX_Model().to(self.device)
        self.model.load_state_dict(checkpoint['model_state'], strict=False)
        self.model.eval()
        self.threshold = checkpoint.get('threshold', 0.001)

    def get_error(self, cmd):
        # Apply the EXACT same normalization as training
        sanitized = normalize_command(cmd)
        inputs = self.tokenizer(sanitized, return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                emb, rec = self.model(inputs['input_ids'], inputs['attention_mask'])
                return torch.mean((emb - rec)**2).item()

    def run_war_room(self, benign_csv="synthetic_benign_baseline.csv", malicious_csv="mitre_atlas_raw.csv"):
        print(f"[*] Running Normalized Performance Sweep (Threshold: {self.threshold:.8f})")
        
        b_df = pd.read_csv(benign_csv).sample(1000)
        m_df = pd.read_csv(malicious_csv)
        
        b_errors = [self.get_error(c) for c in tqdm(b_df['command'], desc="Benign Skeletons")]
        m_errors = [self.get_error(c) for c in tqdm(m_df['command'], desc="Malicious Skeletons")]
        
        b_max, m_min = np.max(b_errors), np.min(m_errors)
        gap = m_min / b_max if b_max > 0 else 0
        
        print(f"\n" + "="*50)
        print(f"FINAL NORMALIZED GAP REPORT")
        print(f"="*50)
        print(f"Max Benign Noise   : {b_max:.8f}")
        print(f"Min Malicious Sign : {m_min:.8f}")
        print(f"THE SAFETY GAP     : {gap:.2f}x")
        print(f"Detection Rate     : {(np.array(m_errors) > self.threshold).mean()*100:.2f}%")
        print(f"False Positive Rate: {(np.array(b_errors) > self.threshold).mean()*100:.2f}%")
        print(f"="*50)

if __name__ == "__main__":
    evaluator = NormalizedEvaluator()
    evaluator.run_war_room()
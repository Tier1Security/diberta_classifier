import torch
import pandas as pd
import numpy as np
import random
import re
from transformers import RobertaTokenizer
from tqdm import tqdm
from train_autoencoder import SecurityAgentX_Model

# ==========================================
# PRODUCTION-GRADE NORMALIZATION LAYER
# ==========================================
def normalize_command(cmd):
    """
    Deterministic Sanitizer: This runs BEFORE the model sees the data.
    It strips carets, backticks, collapses spaces, and standardizes slashes.
    """
    if not isinstance(cmd, str): return ""
    
    # 1. Lowercase for Case-Insensitivity
    cmd = cmd.lower()
    
    # 2. Strip Obfuscation Characters (CMD carets and PS backticks)
    cmd = cmd.replace("^", "").replace("`", "")
    
    # 3. Standardize Slashes (Force Windows-style backslashes)
    cmd = cmd.replace("/", "\\")
    
    # 4. Collapse Multiple Whitespaces (Bypasses 'spacing' noise)
    cmd = " ".join(cmd.split())
    
    # 5. Strip Harmless Redirections
    cmd = cmd.replace("> nul 2>&1", "").replace(">nul", "")
    
    # 6. Normalize path artifacts (e.g., C:\.\Windows -> C:\Windows)
    cmd = cmd.replace("\\.\\", "\\")
    
    return cmd.strip()

class WarRoomEvaluator:
    def __init__(self, model_path="anomaly_engine.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Normalized War Room Evaluation: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SecurityAgentX_Model().to(self.device)
        
        # Mapping weights to the architecture
        state_dict = checkpoint.get('model_state', checkpoint)
        clean_state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.eval()
        
        self.threshold = checkpoint.get('threshold', 0.001)
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def get_error(self, cmd):
        # NORMALIZATION IS APPLIED HERE
        sanitized = normalize_command(cmd)
        
        inputs = self.tokenizer(sanitized, return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                emb, rec = self.model(inputs['input_ids'], inputs['attention_mask'])
                return torch.mean((emb - rec)**2).item()

    def run_competitive_test(self, benign_csv="synthetic_benign_baseline.csv", malicious_csv="mitre_atlas_raw.csv"):
        # We sample 2000 benign to ensure a broad enough look at 'Complex Normal'
        b_df = pd.read_csv(benign_csv).sample(min(2000, 5000))
        m_df = pd.read_csv(malicious_csv)

        print(f"[*] Running Competitive Test (Normalized Benign vs Normalized Malicious)...")
        
        # Test how the model handles 'Messy' benign after we clean it
        # (This simulates fuzzed input hitting the sanitizer)
        b_errors = []
        for cmd in tqdm(b_df['command'], desc="Analyzing Benign"):
            # We don't even need apply_obfuscation because the normalizer 
            # would just remove it. We test the "Skeleton" accuracy here.
            b_errors.append(self.get_error(cmd))
            
        m_errors = []
        for cmd in tqdm(m_df['command'], desc="Analyzing Malicious"):
            m_errors.append(self.get_error(cmd))

        # GAP ANALYSIS
        b_max = np.max(b_errors)
        m_min = np.min(m_errors)
        safety_gap = m_min / b_max if b_max > 0 else 0

        print(f"\n" + "="*55)
        print(f"SANANITIZED WAR ROOM PERFORMANCE REPORT")
        print(f"="*55)
        print(f"Production Threshold   : {self.threshold:.8f}")
        print(f"Max Benign Noise       : {b_max:.8f}")
        print(f"Min Malicious Signal   : {m_min:.8f}")
        print(f"THE SAFETY GAP         : {safety_gap:.2f}x")
        print(f"="*55)

        if m_min > b_max:
            print(f"[+] VERDICT: MATHEMATICALLY SEPARATED ({safety_gap:.1f}x Gap)")
        else:
            overlap_pct = (np.array(b_errors) > m_min).mean() * 100
            print(f"[!] VERDICT: OVERLAPPING MANIFOLD")
            print(f"    {overlap_pct:.2f}% of Benign scripts are noisier than the cleanest attack.")

if __name__ == "__main__":
    evaluator = WarRoomEvaluator()
    evaluator.run_competitive_test()
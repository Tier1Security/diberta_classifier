import torch
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
from tqdm import tqdm
import os
import argparse

# Import the synchronized model class from your training script
try:
    from train_autoencoder import SecurityAgentX_Model
except ImportError:
    print("[!] Error: Could not find 'train_autoencoder.py' in the current directory.")
    print("[!] Ensure your training script is named exactly 'train_autoencoder.py'.")
    exit()

class HeuristicEvaluator:
    def __init__(self, model_path="anomaly_engine.pt", benign_csv="synthetic_benign_baseline.csv"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[!] {model_path} not found.")
            
        print(f"[*] Loading model for evaluation: {model_path} on {self.device}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.model = SecurityAgentX_Model().to(self.device)
        
        # Determine the raw state_dict
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        # --- ROBUST WEIGHT LOADING (Fixes _orig_mod. error) ---
        # Strip prefixes added by torch.compile or DistributedDataParallel
        clean_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("_orig_mod.", "").replace("module.", "")
            clean_state_dict[name] = v
        
        # Load weights into the model
        try:
            self.model.load_state_dict(clean_state_dict)
            print("[+] Weights successfully mapped and loaded.")
        except RuntimeError as e:
            print(f"[!] Strict load failed, attempting non-strict load: {e}")
            self.model.load_state_dict(clean_state_dict, strict=False)

        self.model.eval()
        self.benign_csv = benign_csv

    def get_error(self, cmd):
        """Calculates MSE for a single command string."""
        inputs = self.tokenizer(
            str(cmd).lower(), 
            return_tensors="pt", 
            truncation=True, 
            padding="max_length", 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                emb, rec = self.model(inputs['input_ids'], inputs['attention_mask'])
                return torch.mean((emb - rec)**2).item()

    def run_sweep(self, malicious_csv="mitre_atlas_raw.csv"):
        """
        Sweeps through Z-scores to find the optimal threshold for the 1% FPR limit.
        """
        print(f"[*] Starting Z-Score Sensitivity Sweep...")
        
        if not os.path.exists(self.benign_csv):
            print(f"[!] {self.benign_csv} not found.")
            return

        # Collect raw errors
        benign_df = pd.read_csv(self.benign_csv).sample(min(2000, 5000))
        b_errors = np.array([self.get_error(c) for c in tqdm(benign_df['command'], desc="Scanning Benign")])
        
        if not os.path.exists(malicious_csv):
            print(f"[!] {malicious_csv} not found.")
            return

        mal_df = pd.read_csv(malicious_csv)
        m_errors = np.array([self.get_error(c) for c in tqdm(mal_df['command'], desc="Scanning Malicious")])
        
        mean_b = np.mean(b_errors)
        std_b = np.std(b_errors)
        
        print(f"\n| Z-Score | FPR (%) | Detection (%) | Status |")
        print(f"|---------|---------|---------------|--------|")
        
        best_z = 3.0
        for z in np.arange(0.5, 4.1, 0.5):
            threshold = mean_b + (z * std_b)
            fpr = (b_errors > threshold).mean() * 100
            det = (m_errors > threshold).mean() * 100
            
            status = "PASS" if fpr <= 1.0 else "TOO NOISY"
            print(f"| {z:.1f}     | {fpr:5.2f}% | {det:11.2f}% | {status}   |")
            
            if fpr <= 1.0:
                best_z = z

        print(f"\n[+] Recommended Z-Score for your model: {best_z}")
        print(f"[i] Run with: python3 eval.py --zscore {best_z}")

    def evaluate(self, z_score, malicious_csv="mitre_atlas_raw.csv"):
        benign_df = pd.read_csv(self.benign_csv).sample(min(len(pd.read_csv(self.benign_csv)), 2000))
        b_errors = np.array([self.get_error(c) for c in tqdm(benign_df['command'], desc="Establishing Baseline")])
        threshold = np.mean(b_errors) + (z_score * np.std(b_errors))
        
        print(f"\n--- Performance Analysis (Z={z_score}) ---")
        print(f"Operational Threshold: {threshold:.8f}")
        
        fpr = (b_errors > threshold).mean() * 100
        print(f"[+] False Positive Rate: {fpr:.2f}% (Target < 1.0%)")
        
        if os.path.exists(malicious_csv):
            mal_df = pd.read_csv(malicious_csv)
            m_errors = np.array([self.get_error(c) for c in tqdm(mal_df['command'], desc="Detecting Anomalies")])
            detection_rate = (m_errors > threshold).mean() * 100
            print(f"[+] Detection Rate: {detection_rate:.2f}% (Target > 95.0%)")
            print(f"[+] Signal-to-Noise: {(np.mean(m_errors)/np.mean(b_errors)):.2f}x")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anomaly_engine.pt")
    parser.add_argument("--benign", default="synthetic_benign_baseline.csv")
    parser.add_argument("--malicious", default="mitre_atlas_raw.csv")
    parser.add_argument("--zscore", type=float, default=3.0)
    parser.add_argument("--sweep", action="store_true", help="Find the best Z-score automatically")
    args = parser.parse_args()

    try:
        evaluator = HeuristicEvaluator(model_path=args.model, benign_csv=args.benign)
        if args.sweep:
            evaluator.run_sweep(malicious_csv=args.malicious)
        else:
            evaluator.evaluate(z_score=args.zscore, malicious_csv=args.malicious)
    except Exception as e:
        print(f"[!] Eval Error: {e}")
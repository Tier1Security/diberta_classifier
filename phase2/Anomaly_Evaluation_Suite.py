import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import glob
from train_autoencoder import RobertaAutoencoder # Ensure this matches your training file

# ==========================================
# 1. Automated Merger Utility
# ==========================================
def merge_and_load_data(pattern="benign_baseline_*.csv", output_file="benign_baseline.csv"):
    """Merges all machine baselines into one master CSV."""
    files = glob.glob(pattern)
    if not files:
        print(f"[!] No baseline files found for pattern: {pattern}")
        return None

    print(f"[*] Merging {len(files)} machine baselines...")
    dfs = [pd.read_csv(f) for f in files]
    master_df = pd.concat(dfs).drop_duplicates(subset=['command'])
    master_df.to_csv(output_file, index=False)
    print(f"[+] Master Baseline: {len(master_df)} unique commands.")
    return master_df

# ==========================================
# 2. Evaluation & Metric Engine
# ==========================================
class AnomalyEvaluator:
    def __init__(self, model_path, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=self.device)

        self.model = RobertaAutoencoder().to(self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.threshold = checkpoint['threshold']
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self.model.eval()
        print(f"[+] Model loaded. Operational Threshold: {self.threshold:.6f}")

    def calculate_error(self, command):
        """Calculates reconstruction error for a single command."""
        encoding = self.tokenizer.encode_plus(
            command.lower(),
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            emb, rec = self.model(encoding['input_ids'], encoding['attention_mask'])
            error = torch.mean((emb - rec)**2).item()
        return error

    def run_stress_test(self, malicious_csv="mitre_atlas_raw.csv"):
        """
        Stress tests the engine against the entire MITRE Atlas.
        Checks if malicious commands consistently exceed the benign threshold.
        """
        print(f"\n[*] Starting Adversarial Stress Test against {malicious_csv}...")
        mal_data = pd.read_csv(malicious_csv)

        results = []
        for cmd in tqdm(mal_data['command']):
            error = self.calculate_error(cmd)
            results.append(error)

        results = np.array(results)
        success_rate = (results > self.threshold).mean() * 100

        print(f"\n--- STRESS TEST RESULTS ---")
        print(f"Total TTPs Tested: {len(results)}")
        print(f"Average Malicious Error: {results.mean():.6f}")
        print(f"Threshold Coverage: {success_rate:.2f}% (TTPs correctly flagged as Anomalous)")
        print(f"Max Malicious Error: {results.max():.6f}")
        print(f"Min Malicious Error: {results.min():.6f}")

        return success_rate

    def evaluate_false_positives(self, benign_csv="benign_baseline.csv", sample_size=1000):
        """Verifies False Positive Rate on a held-out benign sample."""
        print(f"\n[*] Evaluating False Positive Rate (FPR)...")
        benign_data = pd.read_csv(benign_csv).sample(sample_size)

        errors = []
        for cmd in tqdm(benign_data['command']):
            errors.append(self.calculate_error(cmd))

        errors = np.array(errors)
        fpr = (errors > self.threshold).mean() * 100
        print(f"False Positive Rate: {fpr:.2f}%")
        return fpr

# ==========================================
# 3. Execution Script
# ==========================================
if __name__ == "__main__":
    # 1. Merge the baselines from your collection machines
    merge_and_load_data()

    # 2. Initialize Evaluator (Assumes you've run train_autoencoder.py)
    try:
        evaluator = AnomalyEvaluator("anomaly_engine.pt")

        # 3. Evaluate FP Rate on Benign Data
        evaluator.evaluate_false_positives()

        # 4. Stress Test against the MITRE Catalogue (The Atlas)
        evaluator.run_stress_test("mitre_atlas_raw.csv")

    except FileNotFoundError:
        print("[!] Missing anomaly_engine.pt or mitre_atlas_raw.csv. Train and Scrape first!")

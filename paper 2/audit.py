import torch
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer
from tqdm import tqdm
import os
import argparse

try:
    from train_autoencoder import SecurityAgentX_Model
except ImportError:
    print("[!] Error: train_autoencoder.py must be in the same directory as this script.")
    exit()

# Ensure this matches your production agent's normalization logic
def normalize_command(cmd):
    if not isinstance(cmd, str): return ""
    cmd = cmd.lower().replace("^", "").replace("`", "").replace("/", "\\")
    return " ".join(cmd.split()).strip()

class AnomalyAuditor:
    def __init__(self, model_path="anomaly_engine.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Resolve path: If not found in CWD, check script directory
        if not os.path.exists(model_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_path = os.path.join(script_dir, model_path)
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"Could not find {model_path} in CWD or script directory.")

        print(f"[*] Loading Engine for Audit: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model = SecurityAgentX_Model().to(self.device)
        
        state_dict = checkpoint.get('model_state', checkpoint)
        clean_state_dict = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(clean_state_dict, strict=False)
        self.model.eval()
        self.threshold = checkpoint.get('threshold', 0.001)
        self.tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def get_error(self, cmd):
        sanitized = normalize_command(cmd)
        inputs = self.tokenizer(sanitized, return_tensors="pt", truncation=True, padding="max_length", max_length=96).to(self.device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=self.dtype):
                emb, rec = self.model(inputs['input_ids'], inputs['attention_mask'])
                return torch.mean((emb - rec)**2).item()

    def audit_mitre_techniques(self, malicious_csv="mitre_atlas_raw.csv"):
        if not os.path.exists(malicious_csv):
            # Check script directory for CSV as well
            script_dir = os.path.dirname(os.path.abspath(__file__))
            malicious_csv = os.path.join(script_dir, malicious_csv)
            if not os.path.exists(malicious_csv):
                print(f"[!] Error: {malicious_csv} not found.")
                return

        df = pd.read_csv(malicious_csv)
        print(f"[*] Auditing {len(df)} commands across unique MITRE techniques...")
        
        audit_results = []
        
        for mitre_id, group in tqdm(df.groupby('mitre_id')):
            errors = []
            worst_cmd = ""
            min_err = float('inf')
            
            for cmd in group['command']:
                err = self.get_error(cmd)
                errors.append(err)
                if err < min_err:
                    min_err = err
                    worst_cmd = cmd
            
            avg_err = np.mean(errors)
            detection_status = "DETECTED" if min_err > self.threshold else "BYPASSED"
            
            audit_results.append({
                "mitre_id": mitre_id,
                "avg_error": avg_err,
                "min_error": min_err,
                "detection": detection_status,
                "worst_offender": worst_cmd
            })
            
        audit_df = pd.DataFrame(audit_results).sort_values(by="min_error")
        
        print("\n" + "="*80)
        print(f"{'MITRE ID':<12} | {'AVG ERROR':<12} | {'MIN ERROR':<12} | {'STATUS'}")
        print("-" * 80)
        
        for _, row in audit_df.head(20).iterrows():
            status_color = "[!]" if row['min_error'] < self.threshold else "[+]"
            print(f"{status_color} {row['mitre_id']:<9} | {row['avg_error']:.8f} | {row['min_error']:.8f} | {row['detection']}")
        
        print("="*80)
        
        weak_links = audit_df[audit_df['min_error'] < self.threshold * 1.5]
        weak_links.to_csv("weak_mitre_links.csv", index=False)
        print(f"[+] Audit complete. Detailed 'Weak Links' saved to weak_mitre_links.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anomaly_engine.pt", help="Path to the model file")
    parser.add_argument("--malicious", default="mitre_atlas_raw.csv", help="Path to malicious TTP CSV")
    args = parser.parse_args()

    try:
        auditor = AnomalyAuditor(model_path=args.model)
        auditor.audit_mitre_techniques(malicious_csv=args.malicious)
    except FileNotFoundError as e:
        print(f"[!] Error: {e}")
        print("[i] Try running the script from within the phase2 folder or provide the full path using --model")
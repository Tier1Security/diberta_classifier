import json
import random
import requests
import pandas as pd
from stix2 import MemoryStore, Filter
from os import path

# --- CONFIG ---
STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
OUTPUT_FILE = "mitre_training_data.jsonl"

def generate_synthetic_log(procedure_text, process_name):
    """
    Simple augmentation to make the model robust to different log formats.
    """
    # Basic raw text (as seen in STIX)
    logs = [procedure_text]
    
    # Simulation 1: Structured Log
    logs.append(f"Process: {process_name} | Command: {procedure_text}")
    
    # Simulation 2: SIEM-style
    logs.append(f"EventID: 4688 | New Process Name: {process_name} | CommandLine: {procedure_text}")
    
    return random.choice(logs)

print("1. Downloading MITRE STIX Data...")
if not path.exists("enterprise-attack.json"):
    r = requests.get(STIX_URL)
    with open("enterprise-attack.json", 'wb') as f:
        f.write(r.content)

print("2. Parsing Relationships...")
with open("enterprise-attack.json", "r") as f:
    stix_json = json.load(f)

mem = MemoryStore(stix_json)
techniques = mem.query([Filter("type", "=", "attack-pattern")])
relationships = mem.query([Filter("type", "=", "relationship"), Filter("relationship_type", "=", "uses")])
tech_map = {t.id: t for t in techniques}

training_pairs = []

for r in relationships:
    if r.target_ref in tech_map:
        technique = tech_map[r.target_ref]
        
        # We only want relationships that have a specific command description
        if hasattr(r, "description"):
            procedure_text = r.description
            
            # Attempt to guess the process name from the text (simple heuristic)
            process = "unknown.exe"
            common_bins = ["cmd.exe", "powershell.exe", "reg.exe", "certutil.exe", "rundll32.exe"]
            for bin in common_bins:
                if bin in procedure_text.lower():
                    process = bin
                    break

            # Create the Training Pair
            # Input (Anchor): The specific command log simulation
            # Output (Positive): The generic MITRE definition
            anchor = generate_synthetic_log(procedure_text, process)
            positive = f"{technique.name}: {technique.description}"
            
            training_pairs.append({"anchor": anchor, "positive": positive, "t_code": technique.external_references[0].external_id})

# Save to JSONL
df = pd.DataFrame(training_pairs)
df.to_json(OUTPUT_FILE, orient="records", lines=True)
print(f"Success! Generated {len(df)} training pairs in '{OUTPUT_FILE}'.")
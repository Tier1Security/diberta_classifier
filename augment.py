import csv
import random
import os
import string

# --- CONFIGURATION ---
OUTPUT_FILE = "v13_svchost_batch.csv" # Renamed output file
NUM_SAMPLES = 2000 
HIVES = ["hklm\\sam", "hklm\\security", "hklm\\system"]

# --- UTILS ---
def clean_text(text): return text.lower()
def get_wrapper(payload):
    wrappers = ["powershell -nop -c \"{}\"", "cmd.exe /c \"{}\""]
    return random.choice(wrappers).format(payload)

# --- GENERATORS ---

# 1. T1003: Wrapped Hard Negative (Targets the wrapped payload)
def generate_wrapped_t1003():
    # 90% chance of being the wrapped reg dump
    if random.random() < 0.9:
        payload = f"reg save {random.choice(HIVES)} c:\\temp\\{random.choice(['sec.dat', 'sam.dat'])}"
        cmd = get_wrapper(payload)
        return clean_text(cmd), 1
    # 10% chance of being a simple naked signal 
    else:
        return clean_text(f"rundll32.exe c:\\windows\\system32\\comsvcs.dll, minidump 1234 c:\\temp\\lsass.dmp full"), 1

# 2. BENIGN: Svchost and System Process Contrast (Targets the new failure)
def generate_benign_contrast():
    # Includes the exact failing case and variations to stabilize the Benign boundary
    cmd = random.choice([
        # --- NEW HARD NEGATIVES (Svchost and Winlogon) ---
        "c:\\windows\\system32\\svchost.exe -k localservicenonetworkfirewall -p", 
        "c:\\windows\\system32\\svchost.exe -k networkservice", 
        "c:\\windows\\system32\\winlogon.exe",
        # --- Existing Benign Recon/Wrapper Contrast ---
        "powershell get-process | select-object name", 
        "wmic useraccount get name", 
        "reg query hkcu\\console"
    ])
    return clean_text(cmd), 0


# --- MAIN SCRIPT ---
data = []
# 50/50 split for high contrast density
for _ in range(NUM_SAMPLES // 2):
    data.append(generate_wrapped_t1003())
for _ in range(NUM_SAMPLES // 2):
    data.append(generate_benign_contrast())

random.shuffle(data)

# Write to CSV
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"]) 
    writer.writerows(data)
    
print(f"Created {NUM_SAMPLES} surgical augmentation samples.")
print(f"NEXT STEP: Manually combine '{OUTPUT_FILE}' with your V9 training CSV and retrain.")
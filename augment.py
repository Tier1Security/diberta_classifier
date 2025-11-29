import csv
import random
import os
import string

# --- CONFIGURATION ---
OUTPUT_FILE = "v15_final_fix_batch.csv"
NUM_SAMPLES = 2000 # High density batch size

# --- UTILS ---
def clean_text(text): return text.lower()
def get_wrapper(payload):
    wrappers = ["powershell -nop -c \"{}\"", "cmd.exe /c \"{}\""]
    return random.choice(wrappers).format(payload)

# --- GENERATORS ---

# 1. T1003 Fix (Targets Confusion with T1134)
def generate_t1003_fix():
    # Use the EXACT failing structure: reg save HKLM\SAM
    hive = random.choice(["hklm\\sam", "hklm\\security"])
    path = f"c:\\temp\\{random.choice(['sec.dat', 'sam.dat'])}"
    
    # 70% of this batch is the direct failing case or its core wrapper
    if random.random() < 0.7:
        cmd = f"reg save {hive} {path}"
    else:
        # Wrapped version to secure against wrapper failures
        payload = f"reg save {hive} {path}"
        cmd = get_wrapper(payload)

    return clean_text(cmd), 1

# 2. T1134 Fix (Targets the 'runas' FN)
def generate_t1134_fix():
    cmd = random.choice([
        # Targets FN 3: The exact failing 'runas' command structure
        "runas /user:administrator cmd.exe",
        "runas /user:system powershell.exe",
        
        # Targets the Kerberos TGT failure
        "klist get krbtgt/domain.local",
        
        # Ensures klist purge confidence is maxed out
        "klist purge",
        "sekurlsa::pth /user:admin /ntlm:hash"
    ])
    return clean_text(cmd), 3

# 3. BENIGN Contrast (Targets the confusing FN 3 'runas' shell behavior)
def generate_benign_contrast():
    cmd = random.choice([
        # Benign shell spawning (The model must learn this is safe)
        "start cmd.exe /c whoami",
        "start powershell.exe -windowstyle hidden",
        
        # Benign Recon
        "wmic product get name, version",
        "reg query hklm\\software\\microsoft",
        
        # The exact query command that resulted in low confidence (FN 2)
        "reg query hklm\\sam /v lsabdsize" 
    ])
    return clean_text(cmd), 0


# --- MAIN SCRIPT ---
data = []
# Split data heavily toward the failed TTPs and the Benign contrast
NUM_T1003 = 700 
NUM_T1134 = 700
NUM_BENIGN = 600

for _ in range(NUM_T1003):
    data.append(generate_t1003_fix())
for _ in range(NUM_T1134):
    data.append(generate_t1134_fix())
for _ in range(NUM_BENIGN):
    data.append(generate_benign_contrast())

random.shuffle(data)

# Write to CSV
with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"]) 
    writer.writerows(data)
    
print(f"Created {len(data)} surgical augmentation samples in {OUTPUT_FILE}.")
print(f"NEXT STEP: Manually combine '{OUTPUT_FILE}' with your training CSV and retrain the model.")
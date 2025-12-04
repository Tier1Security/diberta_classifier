import csv
import random
import os
import string

# --- CONFIGURATION ---
DATA_DIR = "data" # New folder for final, non-toxic fix
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# We keep the large volume, as the model needs the context
TOTAL_EXAMPLES = 200000 
TRAIN_COUNT = int(TOTAL_EXAMPLES * 0.8)
VALIDATION_COUNT = int(TOTAL_EXAMPLES * 0.1)
TEST_COUNT = int(TOTAL_EXAMPLES * 0.1)

# --- UTILS ---
def clean_text(text):
    return text.lower()

def random_name(): 
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

def apply_noise_and_wrapper(cmd):
    if random.random() < 0.20:
        cmd = " ".join(cmd.split())
        cmd = cmd.replace(" ", " " * random.randint(1, 3)) 

        if random.random() < 0.5:
            wrappers = ["cmd.exe /c \"{}\"", "powershell -nop -c \"{}\""]
            cmd = random.choice(wrappers).format(cmd)
            
    return cmd

# ==========================================
# CLASS 0a: BENIGN (HARD NEGATIVES - TOXIN REMOVAL)
# *Removing all REG commands from the Hard Negative pool to eliminate conflict.*
# ==========================================
def generate_hard_negative():
    rng = random.random()
    
    # 35% Hard Negative for T1134/WMIC
    if rng < 0.35: 
        cmd = random.choice([
            "wmic product get name, version", 
            "get-help adjusttokenprivileges -full",
            "sc query windefend", 
            "start cmd.exe /c whoami",
            # New Benign Query using non-reg syntax
            "get-registry-value -path hklm\\sam\\domains", 
        ])
    # 35% Benign Registry Backups (PLACEHOLDER VERBS)
    elif rng < 0.7:
        # These are structurally distinct commands that replace the "reg save/export" functionality
        cmd = random.choice([
            "netsh dump > netconfig.txt",
            "backup-utility save registry config", 
            "backup-utility write hklm\\software to c:\\backups\\config.dat",
            "reg query hkcu\\console" # Only safe queries allowed
        ])
    # 30% Benign Dev Servers/Svchost
    else:
        apps = [
            f"python -m http.server 8080",
            f"node server.js --port 3000",
            f"c:\\windows\\system32\\svchost.exe -k networkservice"
        ]
        cmd = random.choice(apps)
        
    return clean_text(apply_noise_and_wrapper(cmd)), 0

# ==========================================
# CLASS 0b: BENIGN (PURE NOISE) - UNCHANGED
# ==========================================
def generate_pure_noise():
    cmd = random.choice([
        "explorer.exe", "tasklist /v", "systeminfo", "whoami /groups", "ipconfig /all", 
        "ping 192.168.1.1", "dir c:\\users", "powershell get-process", "notepad.exe"
    ])
    return clean_text(apply_noise_and_wrapper(cmd)), 0

# ==========================================
# CLASS 1: CREDENTIAL DUMPING (T1003) - RUTHLESS REINFORCEMENT
# *T1003 is the ONLY place 'reg save/export' is allowed in this dataset.*
# ==========================================
def generate_t1003():
    rng = random.random()
    
    # 1. 70% CHANCE: Wrapped or Unwrapped SAM/SECURITY DUMP (The essential fix)
    if rng < 0.7:
        hive = random.choice(["hklm\\sam", "hklm\\security", "hklm\\system"])
        path = f"c:\\windows\\temp\\{random_name()}.save"
        action = random.choice(["save", "export"])
        payload = f"reg {action} {hive} {path}"
        
        # Force a wrapper 50% of the time to break the Benign wrapper bias
        if random.random() < 0.5:
            cmd = apply_noise_and_wrapper(payload)
        else:
            cmd = payload
        
        return clean_text(cmd), 1
    
    # 2. 30% CHANCE: LSASS and Embedded Dumps
    else:
        cmd = random.choice([
            f"powershell -nop -c \"reg save hklm\\sam c:\\temp\\s.dat\"",
            f"cmd.exe /c \"reg save hklm\\security c:\\temp\\s.dat\"",
            f"rundll32.exe c:\\windows\\system32\\comsvcs.dll, minidump 1234 c:\\temp\\lsass.dmp full"
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 1

# ==========================================
# CLASS 2: DEFENSE EVASION / CLASS 3: TOKEN MANIPULATION - UNCHANGED
# ==========================================
def generate_t1562():
    rng = random.random()
    if rng < 0.5:
        rule = random.choice(["reverse shell", "c2 beacon", "exfil data"])
        app = random.choice([r"c:\windows\temp\nc.exe", r"c:\users\public\malware.exe", "any"])
        feat = f"rule: {rule} | app: {app} | dir: outbound | act: allow | port: {random.choice(['4444', '1337', 'tcp 8888'])}"
        return clean_text(apply_noise_and_wrapper(feat)), 2
    else:
        cmd = random.choice([
            "set-mppreference -disablerealtimemonitoring $true",
            "netsh advfirewall firewall add rule name=\"bad\" action=allow protocol=any port=4444",
            "sc stop windefend",
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 2

def generate_t1134():
    rng = random.random()
    if rng < 0.6:
        func = random.choice(["duplicatetokenex", "openprocesstoken", "impersonateloggedonuser", "amsiutils"])
        if random.random() < 0.5:
            sig = f"[dllimport(\"advapi32.dll\")] extern bool {func}(...)"
        else:
            sig = f"$a = [ref].assembly.gettype('system.management.automation.{func}')"
        return clean_text(apply_noise_and_wrapper(sig)), 3
    else:
        cmd = random.choice([
            "runas /user:administrator cmd.exe", "klist get krbtgt/domain.local",
            "klist purge", "sekurlsa::pth", "token::elevate"
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 3

# --- MAIN LOOP ---
def create_dataset(file_path, target_counts_dict):
    print(f"Generating {file_path}...")
    output_dir = os.path.dirname(file_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    data = []
    
    # Map generator IDs to functions
    generators = {
        'T1003': generate_t1003, 'T1562': generate_t1562, 'T1134': generate_t1134,
        'Benign_Hard': generate_hard_negative, 'Benign_Noise': generate_pure_noise,
    }
    
    # Map generator IDs to their final output label (0=Benign, 1, 2, 3 = Malicious)
    label_map = {
        'T1003': 1, 'T1562': 2, 'T1134': 3,
        'Benign_Hard': 0, 'Benign_Noise': 0,
    }

    for gen_id, target_count in target_counts_dict.items():
        if target_count == 0: continue
        
        generator_func = generators[gen_id]
        final_label = label_map[gen_id]
        
        unique_samples = set()
        attempts = 0
        while len(unique_samples) < target_count and attempts < target_count * 10:
            sample, _ = generator_func() 
            unique_samples.add((sample, final_label))
            attempts += 1
        data.extend(list(unique_samples))

    random.shuffle(data)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"]) 
        writer.writerows(data)
    print(f"Created {file_path} with {len(data)} samples.")

if __name__ == "__main__":
    # --- TARGET DISTRIBUTION (Updated for 200k Total) ---
    TOTAL = 200000 
    
    # Adjusted Distribution Ratios: TTPs (70%) / Hard Negatives (20%) / Pure Noise (10%)
    RATIO_TTP = 0.70
    RATIO_HARD = 0.20
    RATIO_NOISE = 0.10
    
    # Total count for each generator type across the entire dataset
    T_PER_MALICIOUS_CLASS = int(TOTAL * RATIO_TTP / 3) # ~46,666 each
    T_HARD_NEGATIVE = int(TOTAL * RATIO_HARD)          # 40,000
    T_PURE_NOISE = int(TOTAL * RATIO_NOISE)             # 20,000
    
    # Helper to calculate the target size for the training set (80%)
    def calc_target(total_samples):
        return int(total_samples * 0.8)
    
    # Defining the target counts for the generators (for the TRAIN set)
    TRAIN_TARGETS = {
        'T1003': calc_target(T_PER_MALICIOUS_CLASS),
        'T1562': calc_target(T_PER_MALICIOUS_CLASS),
        'T1134': calc_target(T_PER_MALICIOUS_CLASS),
        'Benign_Hard': calc_target(T_HARD_NEGATIVE),
        'Benign_Noise': calc_target(T_PURE_NOISE),
    }

    # Defining the target counts for the VALIDATION set (10%)
    def calc_val_target(total_samples):
        return int(total_samples * 0.1)
    
    VAL_TARGETS = {
        'T1003': calc_val_target(T_PER_MALICIOUS_CLASS),
        'T1562': calc_val_target(T_PER_MALICIOUS_CLASS),
        'T1134': calc_val_target(T_PER_MALICIOUS_CLASS),
        'Benign_Hard': calc_val_target(T_HARD_NEGATIVE),
        'Benign_Noise': calc_val_target(T_PURE_NOISE),
    }
    
    # Defining the target counts for the TEST set (10%)
    TEST_TARGETS = {
        'T1003': calc_val_target(T_PER_MALICIOUS_CLASS),
        'T1562': calc_val_target(T_PER_MALICIOUS_CLASS),
        'T1134': calc_val_target(T_PER_MALICIOUS_CLASS),
        'Benign_Hard': calc_val_target(T_HARD_NEGATIVE),
        'Benign_Noise': calc_val_target(T_PURE_NOISE),
    }

    create_dataset(TRAIN_FILE, TRAIN_TARGETS)
    create_dataset(VALIDATION_FILE, VAL_TARGETS)
    create_dataset(TEST_FILE, TEST_TARGETS)
    print("\n--- HIGH SIGNAL V13 (FINAL DENSITY BOOST) GENERATION COMPLETE ---")
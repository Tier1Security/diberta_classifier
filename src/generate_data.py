import csv
import random
import os
import string

# --- CONFIGURATION ---
DATA_DIR = "data_clean_v9_final" 
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# *** FINAL RECOMMENDED VOLUME INCREASE ***
TOTAL_EXAMPLES = 150000 
TRAIN_COUNT = int(TOTAL_EXAMPLES * 0.8)
VALIDATION_COUNT = int(TOTAL_EXAMPLES * 0.1)
TEST_COUNT = int(TOTAL_EXAMPLES * 0.1)

# --- UTILS ---
def clean_text(text):
    return text.lower()

def random_name(): 
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))

# REINTRODUCED NOISE: Added random spaces/wrappers to 20% of ALL data
def apply_noise_and_wrapper(cmd):
    if random.random() < 0.20:
        # Add random spaces (most common real-world obfuscation)
        cmd = " ".join(cmd.split()) # Normalize existing whitespace
        cmd = cmd.replace(" ", " " * random.randint(1, 3)) 

        # Apply wrapper 50% of the time noise is active
        if random.random() < 0.5:
            wrappers = ["cmd.exe /c \"{}\"", "powershell -nop -c \"{}\""]
            cmd = random.choice(wrappers).format(cmd)
            
    return cmd

# ==========================================
# CLASS 0: BENIGN (Non-Overlapping Keywords)
# ==========================================
def generate_benign():
    rng = random.random()
    
    # 1. Admin Info/Network & Hard Negatives (40%)
    if rng < 0.4:
        cmd = random.choice([
            "ipconfig /all", "ping 8.8.8.8", 
            "whoami /groups", # Safe version of whoami /priv
            "wmic product get name, version", # Switched from Process to Product to break T1134 link
            "sc query windefend", # Service query (Not stop)
            "reg query hkcu\\console", # Use query (safe) instead of save/export
            "get-help adjusttokenprivileges -full", # T1134 Hard Negative
            "select-string -path c:\\logs\\app.log -pattern 'failed logon'", # Safe logging
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 0

    # 2. Benign Registry Backups (NO REG EXPORT/SAVE) - 30%
    elif rng < 0.7:
        # Saving Console Settings and other backups are now explicitly non-'reg save/export'
        cmd = random.choice([
            "netsh dump > netconfig.txt",
            "backup-utility save registry config", # Replaced 'reg save' scenario
            "backup-utility write hklm\\software to c:\\backups\\config.dat"
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 0

    # 3. Benign Dev Servers (CLI format) - 30%
    else:
        apps = [
            f"python -m http.server 8080",
            f"node server.js --port 3000",
            f"java -jar application.jar",
        ]
        return clean_text(apply_noise_and_wrapper(random.choice(apps))), 0

# ==========================================
# CLASS 1: CREDENTIAL DUMPING (T1003) - CORE SIGNAL AMPLIFICATION
# ==========================================
def generate_t1003():
    rng = random.random()
    
    # 1. Direct Registry Dump (40% - Essential signal)
    if rng < 0.4:
        hive = random.choice(["hklm\\sam", "hklm\\security", "hklm\\system"])
        path = f"c:\\windows\\temp\\{random_name()}.save"
        # Crucial: Use both verbs here to make the HIVE the primary signal
        action = random.choice(["save", "export"])
        cmd = f"reg {action} {hive} {path}"
        return clean_text(apply_noise_and_wrapper(cmd)), 1
    
    # 2. LSASS/Wrapped Registry Hard Negative (60% - Test robustness)
    else:
        # Ensures that a large percentage of T1003 training data uses wrappers
        hive = random.choice(["hklm\\sam", "hklm\\security"])
        payload = f"reg {random.choice(['save', 'export'])} {hive} c:\\temp\\{random_name()}.dat"
        
        cmd = random.choice([
            f"powershell -nop -c \"{payload}\"",
            f"cmd.exe /c \"{payload}\"",
            f"rundll32.exe c:\\windows\\system32\\comsvcs.dll, minidump 1234 c:\\temp\\lsass.dmp full"
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 1

# ==========================================
# CLASS 2: DEFENSE EVASION / FIREWALL (T1562) - UNCHANGED
# ==========================================
def generate_t1562():
    rng = random.random()
    
    # 1. Malicious Firewall LOGS (Focus on external signals/ports)
    if rng < 0.5:
        rule = random.choice(["reverse shell", "c2 beacon", "exfil data"])
        app = random.choice([r"c:\windows\temp\nc.exe", r"c:\users\public\malware.exe", "any"])
        feat = f"rule: {rule} | app: {app} | dir: outbound | act: allow | port: {random.choice(['4444', '1337', 'tcp 8888'])}"
        return clean_text(apply_noise_and_wrapper(feat)), 2
        
    # 2. Defense Evasion Commands (Verbs are the signal: STOP, DISABLE)
    else:
        cmd = random.choice([
            "set-mppreference -disablerealtimemonitoring $true",
            "netsh advfirewall firewall add rule name=\"bad\" action=allow protocol=any port=4444",
            "sc stop windefend",
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 2

# ==========================================
# CLASS 3: TOKEN MANIPULATION (T1134) - REINFORCED APIS
# ==========================================
def generate_t1134():
    # 1. C# Reflection/PowerShell Bypass (60%)
    if random.random() < 0.6:
        func = random.choice([
            "duplicatetokenex", "openprocesstoken", "impersonateloggedonuser", "amsiutils" 
        ])
        
        if random.random() < 0.5:
            sig = f"[dllimport(\"advapi32.dll\")] extern bool {func}(...)"
        else:
            sig = f"$a = [ref].assembly.gettype('system.management.automation.{func}')"
            
        return clean_text(apply_noise_and_wrapper(sig)), 3
        
    # 2. Explicit Token Theft/Purge (40%)
    else:
        cmd = random.choice([
            "klist purge",
            "sekurlsa::pth",
            "token::elevate",
        ])
        return clean_text(apply_noise_and_wrapper(cmd)), 3

# --- MAIN LOOP (unchanged) ---
def create_dataset(file_path, target_per_class):
    print(f"Generating {file_path}...")
    output_dir = os.path.dirname(file_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    data = []
    generators = {
        0: generate_benign,
        1: generate_t1003,
        2: generate_t1562,
        3: generate_t1134
    }
    
    for label, generator_func in generators.items():
        unique_samples = set()
        attempts = 0
        while len(unique_samples) < target_per_class and attempts < target_per_class * 10:
            sample, lbl = generator_func()
            unique_samples.add((sample, lbl))
            attempts += 1
        data.extend(list(unique_samples))

    random.shuffle(data)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"]) 
        writer.writerows(data)
    print(f"Created {file_path} with {len(data)} samples.")

if __name__ == "__main__":
    create_dataset(TRAIN_FILE, TRAIN_COUNT // 4)
    create_dataset(VALIDATION_FILE, VALIDATION_COUNT // 4)
    create_dataset(TEST_FILE, TEST_COUNT // 4)
    print("\n--- HIGH SIGNAL V9 GENERATION COMPLETE ---")
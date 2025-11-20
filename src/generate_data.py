import csv
import random
import os

# --- HELPER FUNCTIONS ---
def normalize_case(text):
    return text.lower()

def random_space():
    return ' ' * random.randint(1, 3)

# --- CONFIGURATION ---
DATA_DIR = "data_3class"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

TOTAL_EXAMPLES = 20000 # Increased slightly to accommodate 3 classes
TRAIN_COUNT = int(TOTAL_EXAMPLES * 0.8)
VALIDATION_COUNT = int(TOTAL_EXAMPLES * 0.1)
TEST_COUNT = int(TOTAL_EXAMPLES * 0.1)

# --- 1. T1003.002 COMPONENTS (SAM - Label 1) ---
# Kept exactly as is from your previous logic
SAM_COMPONENTS = {
    "executables": ["reg.exe", "reg"],
    "actions": ["save", "export"],
    "hives": [
        "hklm\\sam", "hklm\\system", "hklm\\security",
        "HKEY_LOCAL_MACHINE\\SAM", "HKEY_LOCAL_MACHINE\\SYSTEM", "HKEY_LOCAL_MACHINE\\SECURITY"
    ],
    "paths": [
        "C:\\Windows\\Temp\\sam.save", "C:\\Temp\\system.hive", "%TEMP%\\sec_backup.dat",
        "\\\\localhost\\c$\\__dump\\sam.bak", "C:\\Users\\Public\\registry_export.reg", "\\.\\C$\backup.reg"
    ],
    "wrappers": [
        "", "cmd.exe /c {}", "powershell -command {}", "CMD /C powershell -c \"{}\"",
        "Pwsh -noprofile -c {}", "C:\\Windows\\system32\\cmd.exe /k {}",
        "powershell -exec bypass -w hidden -c {}", "cmd.exe /c start /b {}"
    ]
}

# --- 2. T1134 COMPONENTS (Token Manipulation - Label 2) ---
# New components based on your image and T1134 generic tools
TOKEN_COMPONENTS = {
    "scripts": [
        "Enable-Privilege.ps1", "EnableAllTokenPrivs.ps1", "Set-TokenPrivs.ps1", 
        "Invoke-TokenManipulation.ps1", "AdjustToken.ps1"
    ],
    "commands": [
        "Import-Module .\\{}", "Import-Module {}", ".\\{}", "IEX (New-Object Net.WebClient).DownloadString('{}')"
    ],
    "privileges": [
        "SeTakeOwnershipPrivilege", "SeDebugPrivilege", "SeRestorePrivilege", 
        "SeBackupPrivilege", "SeTcbPrivilege"
    ],
    "native_cmds": [
        "whoami /priv", "whoami /groups", # Contextual discovery often run with token manipulation
        "cmd.exe /c whoami /priv", 
    ],
    "wrappers": [
        "powershell -ep bypass {}", "powershell -c {}", "pwsh {}", "cmd /c powershell {}"
    ]
}

# --- 3. BENIGN COMPONENTS (Label 0) ---
# Updated to include benign PowerShell to confuse T1134
BENIGN = {
    # --- Registry Components (Existing) ---
    "reg_executables": ["reg.exe", "reg"],
    "reg_actions": ["query", "add", "delete", "copy", "compare", "restore", "unload", "save"], 
    "hives": ["hkcu", "hku", "HKEY_CURRENT_USER", "HKEY_USERS", "HKEY_LOCAL_MACHINE"],
    "keys": [
        "Software\\Microsoft\\Windows\\CurrentVersion", "Software\\Google\\Chrome",
        "System\\CurrentControlSet\\Services", "Control Panel\\Desktop", "Software\\Policies\\MyCorp"
    ],
    "value_names": ["/v version", "/v path", "/v lastupdate", "/f", ""],
    "benign_save_paths": ["C:\\temp\\user_settings.reg", "C:\\Users\\Public\\chrome_data.bak"],
    
    # --- New Benign PowerShell Components (To counter T1134 FPs) ---
    "ps_modules": [
        "ActiveDirectory", "Hyper-V", "NetAdapter", "DnsClient", "BitLocker", "ConfigManager"
    ],
    "ps_scripts": [
        "Backup-Daily.ps1", "Update-System.ps1", "Check-DiskSpace.ps1", "Log-UserActivity.ps1"
    ],
    "ps_cmds": [
        "Get-Service", "Get-Process", "Get-Date", "whoami /user", "whoami /all", "hostname"
    ],
    "wrappers": ["", "cmd /c {}", "powershell -c {}", "start /b {}"]
}

# --- GENERATION FUNCTIONS ---

def generate_sam_command():
    """Generates T1003.002 (Label 1)"""
    comp = SAM_COMPONENTS
    exe = random.choice(comp["executables"])
    action = random.choice(comp["actions"])
    hive = random.choice(comp["hives"])
    path = random.choice(comp["paths"])
    flags = random.choice(["", "/y"]) 
    
    cmd = f"{exe}{random_space()}{action}{random_space()}{hive}{random_space()}{path}{random_space()}{flags}".strip()
    
    wrapper = random.choice(comp['wrappers'])
    if wrapper:
        cmd = wrapper.format(cmd)
    
    return normalize_case(cmd), 1

def generate_token_command():
    """Generates T1134 (Label 2)"""
    comp = TOKEN_COMPONENTS
    
    subtype = random.choice(["script_exec", "priv_check", "priv_enable"])
    
    if subtype == "script_exec":
        # e.g., Import-Module .\Enable-Privilege.ps1
        script = random.choice(comp["scripts"])
        base_cmd = random.choice(comp["commands"]).format(script)
    
    elif subtype == "priv_enable":
        # e.g., Manually setting privileges via PS
        priv = random.choice(comp["privileges"])
        base_cmd = f"Get-Process | Select-Object -Property ProcessName, @{{Name='{priv}';Expression={{$_.PrivilegeState}}}}"
        
    else: 
        # e.g., whoami /priv
        base_cmd = random.choice(comp["native_cmds"])

    # Wrap it 50% of the time
    if random.random() > 0.5:
        wrapper = random.choice(comp["wrappers"])
        final_cmd = wrapper.format(base_cmd)
    else:
        final_cmd = base_cmd

    return normalize_case(final_cmd), 2

def generate_benign_command():
    """Generates Benign (Label 0) - Mixed Registry and PowerShell"""
    comp = BENIGN
    
    # 50/50 split between Benign Registry (Anti-SAM) and Benign PowerShell (Anti-Token)
    if random.random() < 0.5:
        # --- Benign Registry Logic ---
        exe = random.choice(comp["reg_executables"])
        action = random.choice(comp["reg_actions"])
        hive = random.choice(comp["hives"])
        key = random.choice(comp["keys"])
        
        if action == "save":
             # Safe save logic
             path = random.choice(comp["benign_save_paths"])
             cmd = f'{exe} {action} "{hive}\\{key}" "{path}"'
        else:
             val = random.choice(comp["value_names"])
             cmd = f'{exe} {action} "{hive}\\{key}" {val}'
    else:
        # --- Benign PowerShell Logic ---
        scenario = random.choice(["module", "script", "cmd"])
        if scenario == "module":
            mod = random.choice(comp["ps_modules"])
            cmd = f"Import-Module {mod}"
        elif scenario == "script":
            script = random.choice(comp["ps_scripts"])
            cmd = f".\\{script}"
        else:
            cmd = random.choice(comp["ps_cmds"])

    # Wrap it sometimes
    wrapper = random.choice(comp['wrappers'])
    if wrapper and random.random() < 0.3:
        cmd = wrapper.format(cmd)
        
    return normalize_case(cmd), 0

def create_dataset(file_path, num_per_class):
    """Generates balanced dataset for 3 classes"""
    print(f"Generating {file_path}...")
    
    output_dir = os.path.dirname(file_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    data = []
    
    # 1. Generate Label 1 (SAM)
    for _ in range(num_per_class):
        data.append(generate_sam_command())

    # 2. Generate Label 2 (Token)
    for _ in range(num_per_class):
        data.append(generate_token_command())
        
    # 3. Generate Label 0 (Benign)
    # We generate slightly more benign to cover both Reg and PS scenarios
    for _ in range(num_per_class):
        data.append(generate_benign_command())

    # 4. Inject Hard Negatives (Critical for T1134)
    # Benign uses of 'whoami' or 'privilege' words in non-attack contexts
    hard_negatives = [
        ("whoami /user", 0),
        ("whoami /all", 0),
        ("echo check privileges", 0),
        ("write-host 'checking admin privileges'", 0),
        ("Import-Module ActiveDirectory", 0)
    ]
    for _ in range(int(num_per_class * 0.05)): # 5% injection
         data.extend(hard_negatives)

    random.shuffle(data)
    
    # Dedup
    data = list(set(data))
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"]) 
        writer.writerows(data)
    
    print(f"Created {file_path} with {len(data)} samples.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # We split the total count by 3 (roughly)
    train_per_class = TRAIN_COUNT // 3
    val_per_class = VALIDATION_COUNT // 3
    test_per_class = TEST_COUNT // 3
    
    create_dataset(TRAIN_FILE, train_per_class)
    create_dataset(VALIDATION_FILE, val_per_class)
    create_dataset(TEST_FILE, test_per_class)
    
    print("\n--- Dataset Generation Complete (3 Classes) ---")
    print("Label 0: Benign")
    print("Label 1: T1003.002 (SAM)")
    print("Label 2: T1134 (Token Manipulation)")
import csv
import random
import os

# --- HELPER FUNCTIONS FOR OBFUSCATION ---
def normalize_case(text):
    """Introduces random case variations for better obfuscation training (approx. 70% chance of mixed case)."""
    if random.random() < 0.7:
        # Apply mixed casing
        return ''.join(c.upper() if random.random() < 0.5 else c.lower() for c in text)
    # Default to lowercase
    return text.lower()

def random_space():
    """Returns 1 to 3 spaces for whitespace variation."""
    return ' ' * random.randint(1, 3)

# --- CONFIGURATION ---
DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")
HELD_OUT_FILE = os.path.join(DATA_DIR, "held_out.csv")

TOTAL_EXAMPLES = 10000
TRAIN_COUNT = int(TOTAL_EXAMPLES * 0.8)
VALIDATION_COUNT = int(TOTAL_EXAMPLES * 0.1)
TEST_COUNT = int(TOTAL_EXAMPLES * 0.1)
HELD_OUT_COUNT = 4000 

# --- DATA POOLS ---

# --- Malicious Components (UPDATED WRAPPERS for FN Fix) ---
MALICIOUS = {
    "executables": ["reg.exe", "reg"],
    "actions": ["save", "export"],
    "hives": [
        "hklm\\sam", "hklm\\system", "hklm\\security",
        "HKEY_LOCAL_MACHINE\\SAM", "HKEY_LOCAL_MACHINE\\SYSTEM", "HKEY_LOCAL_MACHINE\\SECURITY"
    ],
    "paths": [
        "C:\\Windows\\Temp\\sam.save",
        "C:\\Temp\\system.hive",
        "%TEMP%\\sec_backup.dat",
        "\\\\localhost\\c$\\__dump\\sam.bak",
        "C:\\Users\\Public\\registry_export.reg",
        "\\.\\C$\backup.reg"
    ],
    "wrappers": [
        "", 
        "cmd.exe /c {}", 
        "powershell -command {}", 
        # New, complex wrappers to specifically train against nested obfuscation
        "CMD /C powershell -c \"{}\"",
        "Pwsh -noprofile -c {}",
        "C:\\Windows\\system32\\cmd.exe /k {}",
        "powershell -exec bypass -w hidden -c {}",
        "cmd.exe /c start /b {}"
    ]
}

# --- Benign Components (UPDATED WRAPPERS) ---
BENIGN = {
    "executables": ["reg.exe", "reg"],
    "actions": ["query", "add", "delete", "copy", "compare", "restore", "unload", "save"], 
    "hives": ["hkcu", "hku", "HKEY_CURRENT_USER", "HKEY_USERS", "HKEY_LOCAL_MACHINE"],
    "keys": [
        "Software\\Microsoft\\Windows\\CurrentVersion",
        "Software\\Google\\Chrome",
        "System\\CurrentControlSet\\Services",
        "Control Panel\\Desktop",
        "Environment",
        "Software\\MyApp\\Settings",
        "Software\\Policies\\MyCorp",
        "HKEY_LOCAL_MACHINE\\Software\\Policies\\Microsoft", 
        "HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Control", 
        "HKEY_LOCAL_MACHINE\\Software\\Wow6432Node",
        "Software\\Classes"
    ],
    "value_names": ["/v version", "/v path", "/v lastupdate", "/f", ""],
    "data_types": ["/t reg_sz", "/t reg_dword", ""],
    "data": ['/d "1.0.0"', '/d 1', '/d "C:\\Program Files\\App"', ''],
    "benign_save_paths": [
        "C:\\temp\\user_settings.reg",
        "C:\\Users\\Public\\chrome_data.bak",
        "C:\\Windows\\Temp\\software_dump.dat",
        "%TEMP%\\non_sensitive.hive",
        "D:\\Backup\\registry_hive.dat"
    ],
    # NEW: Benign wrappers to ensure model doesn't just flag "wrapper = malicious"
    "wrappers": ["", "cmd /c {}", "powershell -c {}", "start /b {}"],
}

# --- CRITICAL NEGATIVE SAMPLES (Simple FPs) ---
CRITICAL_NEGATIVE_SAMPLES = [
    ("cmd.exe", 0), ("powershell.exe", 0), ("reg.exe", 0), ("reg", 0), ("reg.exe query", 0),
    ("reg query", 0), ("reg.exe save", 0), ("reg save", 0), ("wmic.exe", 0), ("tasklist.exe", 0),
]

# --- PERSISTENT FALSE POSITIVES (Hard Benign Saves for FP Fix) ---
# These are legitimate administrative commands that were previously misclassified as malicious.
PERSISTENT_FALSE_POSITIVES = [
    ("cmd /c reg save HKEY_LOCAL_MACHINE\\Software\\Policies C:\\Backup\\pol.reg", 0),
    ("reg save HKEY_LOCAL_MACHINE\\SOFTWARE C:\\temp\\software_full.reg", 0),
    ("reg save HKLM\\Software\\Wow6432Node C:\\temp\\wow64.reg", 0),
    ("powershell -c reg.exe save HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet C:\\temp\\currentcontrolset.dat", 0),
    ("netsh firewall set opmode enable", 0),
    ("attrib -h -s C:\\temp\\my_file.txt", 0),
]


# --- DATA GENERATION FUNCTIONS ---

def generate_malicious_command():
    """Generates a single, random malicious command using complex wrappers and obfuscation."""
    comp = MALICIOUS
    
    # 1. Build the core command
    exe = random.choice(comp["executables"])
    action = random.choice(comp["actions"])
    hive = random.choice(comp["hives"])
    path = random.choice(comp["paths"])
    flags = random.choice(["", "/y"]) 
    
    command_str = f"{exe}{random_space()}{action}{random_space()}{hive}{random_space()}{path}{random_space()}{flags}".strip()
    
    # 2. Wrap it with a complex wrapper
    wrapper = random.choice(comp['wrappers'])
    if wrapper:
        command_str = wrapper.format(command_str)
    
    # 3. Apply final obfuscation
    command_str = normalize_case(command_str)
        
    return command_str, 1

def generate_benign_command():
    """Generates a single, random benign command, including forced HKLM save examples."""
    comp = BENIGN
    
    exe = random.choice(comp["executables"])
    action = random.choice(comp["actions"])
    
    # Force Benign HKLM Save 20% of the time (Benign Counter-Pattern)
    if action == "save" and random.random() < 0.6: 
        hive = "HKEY_LOCAL_MACHINE"
        # Filter for non-sensitive HKLM keys
        safe_hklm_keys = [k for k in comp['keys'] if hive in k and not any(h in k for h in ["SAM", "SYSTEM", "SECURITY"])]
        if safe_hklm_keys:
            key_path = random.choice(safe_hklm_keys)
        else:
            # Fallback to a non-HKLM key if list is empty
            key_path = random.choice([k for k in comp['keys'] if "HKEY_LOCAL_MACHINE" not in k])
    else:
        hive = random.choice(comp["hives"])
        key_path = random.choice(comp["keys"])
    
    if random.random() > 0.5:
        key_path += "\\" + "SubKey" + str(random.randint(1,100))
    
    # Clean up potential double HKLM pathing
    full_key = f"{hive}\\{key_path}".replace("hkey_local_machine\\hkey_local_machine", "hkey_local_machine", 1)

    space = random_space()
    action_lower = action.lower()
    
    # 2. Construct the core command
    if action_lower == "query":
        val_name = random.choice(comp["value_names"])
        command_str = f'{exe}{space}{action}{space}"{full_key}"{space}{val_name}'.strip()
    elif action_lower == "add":
        data_type = random.choice(comp["data_types"])
        data = random.choice(comp["data"])
        val_name = random.choice(comp["value_names"])
        command_str = f'{exe}{space}{action}{space}"{full_key}"{space}{val_name}{space}{data_type}{space}{data}'.strip()
    elif action_lower == "delete" or action_lower == "unload":
        val_name = random.choice(comp["value_names"])
        command_str = f'{exe}{space}{action}{space}"{full_key}"{space}{val_name}'.strip()
    elif action_lower == "save":
        save_path = random.choice(comp["benign_save_paths"])
        flags = random.choice(["", "/y"]) 
        command_str = f'{exe}{space}{action}{space}"{full_key}"{space}"{save_path}"{space}{flags}'.strip()
    else: # For copy, compare, restore, etc.
        dest_key = f"{hive}\\{random.choice(comp['keys'])}_{random.randint(100,999)}"
        command_str = f'{exe}{space}{action}{space}"{full_key}"{space}"{dest_key}"'.strip()

    # 3. Apply wrapper (50% chance)
    wrapper = random.choice(comp['wrappers'])
    if wrapper and random.random() < 0.5:
        if wrapper:
            command_str = wrapper.format(command_str)
        
    # 4. Apply final obfuscation
    command_str = normalize_case(command_str)

    return command_str, 0

def create_dataset(file_path, num_malicious, num_benign):
    """Creates a balanced and shuffled dataset, injecting critical and hard benign samples."""
    print(f"Generating {file_path} with target counts: {num_malicious} malicious, {num_benign} benign...")
    
    output_dir = os.path.dirname(file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
    data = []
    
    # 1. Generate Malicious Samples
    for _ in range(num_malicious):
        data.append(generate_malicious_command())
        
    # 2. Inject Critical Negative Samples (Simple FPs)
    for command, label in CRITICAL_NEGATIVE_SAMPLES:
        for _ in range(3): # Inject 3 copies of each simple FP
            data.append((command, label))
            
    # 3. Inject Persistent False Positives (Hard Benign Saves - FP Fix)
    for command, label in PERSISTENT_FALSE_POSITIVES:
        for _ in range(10): # Inject 10 copies of each hard benign save
            data.append((command, label))
            
    # 4. Generate Remaining Random Benign Samples
    injected_count = (len(CRITICAL_NEGATIVE_SAMPLES) * 3) + (len(PERSISTENT_FALSE_POSITIVES) * 10)
    remaining_benign = num_benign - injected_count
    if remaining_benign < 0: remaining_benign = 0
    
    for _ in range(remaining_benign):
        data.append(generate_benign_command())
        
    random.shuffle(data)
    
    # Remove duplicates before saving
    data_set = set(data)
    data = list(data_set)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"]) 
        writer.writerows(data)
    print(f"Successfully created {file_path}. Final unique samples: {len(data)}.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- Starting Robust Dataset Generation (10K Total) ---")
    
    # Ensure balanced targets for randomized generation
    train_mal_count = train_ben_count = TRAIN_COUNT // 2
    val_mal_count = val_ben_count = VALIDATION_COUNT // 2
    test_mal_count = test_ben_count = TEST_COUNT // 2
    held_out_mal_count = held_out_ben_count = HELD_OUT_COUNT // 2
    
    create_dataset(TRAIN_FILE, train_mal_count, train_ben_count)
    create_dataset(VALIDATION_FILE, val_mal_count, val_ben_count)
    create_dataset(TEST_FILE, test_mal_count, test_ben_count)
    create_dataset(HELD_OUT_FILE, held_out_mal_count, held_out_ben_count)
    
    print("\n--- Dataset Generation Complete. Ready for training. ---")
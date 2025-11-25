import csv
import random
import os

# --- HELPER FUNCTIONS ---
def normalize_case(text):
    return text.lower()

def random_space():
    return ' ' * random.randint(1, 2)

def maybe_quote(text):
    """Randomly adds quotes to a string (50% chance)."""
    if random.random() < 0.5:
        return f'"{text}"'
    return text

# --- CONFIGURATION ---
DATA_DIR = "data_3class_security"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

TOTAL_EXAMPLES = 25000 
TRAIN_COUNT = int(TOTAL_EXAMPLES * 0.8)
VALIDATION_COUNT = int(TOTAL_EXAMPLES * 0.1)
TEST_COUNT = int(TOTAL_EXAMPLES * 0.1)

# ==========================================
# 1. MALICIOUS REGISTRY (T1003.002) - LABEL 1
# ==========================================
SAM_COMPONENTS = {
    "executables": ["reg.exe", "reg"],
    "actions": ["save", "export"],
    "hives": [
        "hklm\\sam", "hklm\\system", "hklm\\security",
        "HKEY_LOCAL_MACHINE\\SAM", "HKEY_LOCAL_MACHINE\\SYSTEM"
    ],
    "paths": [
        "C:\\Windows\\Temp\\sam.save", "C:\\Temp\\system.hive", 
        "\\\\localhost\\c$\\__dump\\sam.bak", "C:\\Users\\Public\\registry_export.reg",
        "C:\\ProgramData\\backup.hive"
    ],
    "wrappers": ["", "cmd.exe /c {}", "powershell -command {}", "cmd /c start /b {}"]
}

# ==========================================
# 2. MALICIOUS FIREWALL (T1562) - LABEL 2
# ==========================================
FW_MALICIOUS_COMPONENTS = {
    "names": [
        "Allow All", "WinUpdate Helper", "Netcat Listener", "Sliver C2", 
        "Proxy Ingress", "Metasploit", "My Backdoor", "Bypass",
        "Google Chrome Update", "Microsoft Teams Service", "System Integrity"
    ],
    "apps": [
        "C:\\Windows\\Temp\\nc.exe", "C:\\Users\\Public\\updater.exe", 
        "C:\\Temp\\shell.exe", "powershell.exe", "cmd.exe", "rundll32.exe", 
        "C:\\Users\\Public\\chrome.exe", "C:\\ProgramData\\svchost.exe", "C:\\Temp\\teams.exe"
    ],
    "directions": ["Inbound", "Outbound"],
    "actions": ["Allow"], 
    "ports": [
        "4444", "1337", "8888", "9999", "Any", "*",
        "443", "80", "8080", "53" 
    ]
}

# ==========================================
# 3. BENIGN COMPONENTS - LABEL 0
# ==========================================
BENIGN_REG_QUERIES = {
    "executables": ["reg.exe", "reg"],
    "actions": ["query", "compare", "add", "delete", "copy"], 
    "hives": ["hkcu", "hku", "HKEY_CURRENT_USER", "Software\\Microsoft", "HKLM\\Software\\MyCorp"],
    "keys": ["Software\\Microsoft\\Windows\\CurrentVersion", "Control Panel\\Desktop", "Environment"],
    "values": ["/v version", "/v path", "/f", "/s"]
}

BENIGN_REG_SAVES = {
    "executables": ["reg.exe", "reg"],
    "actions": ["save", "export"], 
    "hives": [
        "hkcu\\Software", "HKLM\\Software\\MyCorp", "HKLM\\System\\CurrentControlSet\\Services\\LanmanServer",
        "HKEY_USERS\\.DEFAULT\\Environment", "HKCU\\Console"
    ],
    "paths": [
        "C:\\Backups\\console.reg", "C:\\Temp\\app_config.bkp", 
        "D:\\Logs\\reg_dump.hiv", "\\\\server\\share\\user_profile.dat"
    ]
}
# ==========================================
# 3. BENIGN COMPONENTS - LABEL 0
# FINAL UPDATE: Add "Allow" rules to Benign to fix the Apache False Positive
# ==========================================

BENIGN_FW_SAFE = {
    "names": [
        "Core Networking", "Microsoft Teams", "Spotify Music", "Google Chrome", 
        "Remote Desktop", "Apache Web Server", "Nginx HTTPS", "File Sharing",
        "IIS Worker Process", "Node.js Server", "Python Web App",
        # --- NEW: Neutralize the "Allow" keyword bias ---
        "Allow HTTP", "Allow HTTPS Traffic", "Allow Inbound 80", 
        "Allow Local Subnet", "Allow File Printer Sharing"
    ],
    "apps": [
        "C:\\Program Files\\Microsoft Teams\\current\\Teams.exe",
        "C:\\Windows\\System32\\svchost.exe",
        "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
        "C:\\Apache\\bin\\httpd.exe", 
        "C:\\nginx\\nginx.exe",
        "C:\\Windows\\System32\\inetsrv\\w3wp.exe",
        "C:\\Program Files\\nodejs\\node.exe",
        "C:\\Python39\\python.exe",
        "System"
    ],
    "directions": ["Outbound", "Inbound", "Inbound"], 
    "actions": ["Allow", "Block"],
    "ports": ["80", "443", "53", "123", "3389", "137", "445", "8080", "8443"]
}
BENIGN_ADMIN_TOOLS = [
    "whoami /all", "whoami /user", "ipconfig /all", "net user", "systeminfo", "hostname"
]

# --- GENERATION FUNCTIONS ---

def generate_sam_command():
    """Generates Label 1: Registry Hive Theft"""
    comp = SAM_COMPONENTS
    exe = random.choice(comp["executables"])
    action = random.choice(comp["actions"])
    
    # Randomly quote the hive and path
    hive = maybe_quote(random.choice(comp["hives"]))
    path = maybe_quote(random.choice(comp["paths"]))
    
    flags = random.choice(["", "/y"]) 
    
    cmd = f"{exe}{random_space()}{action}{random_space()}{hive}{random_space()}{path}{random_space()}{flags}".strip()
    
    wrapper = random.choice(comp['wrappers'])
    if wrapper: cmd = wrapper.format(cmd)
    
    return normalize_case(cmd), 1

def generate_firewall_malicious():
    """Generates Label 2: Malicious Firewall Event"""
    comp = FW_MALICIOUS_COMPONENTS
    rule = random.choice(comp["names"])
    app = random.choice(comp["apps"])
    direction = random.choice(comp["directions"])
    action = random.choice(comp["actions"])
    port = random.choice(comp["ports"])
    
    feature_str = f"rule: {rule} | app: {app} | dir: {direction} | act: {action} | port: {port}"
    return normalize_case(feature_str), 2

def generate_benign_mixed():
    """Generates Label 0: Hard Negatives & Normal Activity"""
    rand = random.random()
    
    if rand < 0.3: # 30% SAFE SAVES
        comp = BENIGN_REG_SAVES
        exe = random.choice(comp["executables"])
        action = random.choice(comp["actions"])
        
        # Randomly quote here too!
        hive = maybe_quote(random.choice(comp["hives"]))
        path = maybe_quote(random.choice(comp["paths"]))
        
        cmd = f'{exe} {action} {hive} {path}'
        return normalize_case(cmd), 0

    elif rand < 0.5: # 20% SAFE QUERIES
        comp = BENIGN_REG_QUERIES
        exe = random.choice(comp["executables"])
        action = random.choice(comp["actions"])
        
        # Build path and quote potentially
        hive = random.choice(comp["hives"])
        key = random.choice(comp["keys"])
        full_key = maybe_quote(f"{hive}\\{key}")
        
        val = random.choice(comp["values"])
        cmd = f'{exe} {action} {full_key} {val}'
        return normalize_case(cmd), 0
        
    elif rand < 0.8: # 30% BENIGN FIREWALL
        comp = BENIGN_FW_SAFE
        rule = random.choice(comp["names"])
        app = random.choice(comp["apps"])
        direction = random.choice(comp["directions"])
        action = random.choice(comp["actions"])
        port = random.choice(comp["ports"])
        feature_str = f"rule: {rule} | app: {app} | dir: {direction} | act: {action} | port: {port}"
        return normalize_case(feature_str), 0
        
    else: # 20% BENIGN ADMIN TOOLS
        base_cmd = random.choice(BENIGN_ADMIN_TOOLS)
        wrappers = ["", "cmd /c {}", "cmd.exe /c {}", "powershell -c {}"]
        wrapper = random.choice(wrappers)
        final_cmd = wrapper.format(base_cmd) if wrapper else base_cmd
        return normalize_case(final_cmd), 0

def create_dataset(file_path, num_per_class):
    """Generates balanced dataset for 3 classes"""
    print(f"Generating {file_path}...")
    
    output_dir = os.path.dirname(file_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    data = []
    
    for _ in range(num_per_class): data.append(generate_sam_command())
    for _ in range(num_per_class): data.append(generate_firewall_malicious())
    for _ in range(num_per_class): data.append(generate_benign_mixed())

    random.shuffle(data)
    data = list(set(data))
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"]) 
        writer.writerows(data)
    
    print(f"Created {file_path} with {len(data)} samples.")

if __name__ == "__main__":
    train_per_class = TRAIN_COUNT // 3
    val_per_class = VALIDATION_COUNT // 3
    test_per_class = TEST_COUNT // 3
    create_dataset(TRAIN_FILE, train_per_class)
    create_dataset(VALIDATION_FILE, val_per_class)
    create_dataset(TEST_FILE, test_per_class)
    print("\n--- ROBUST Dataset Generation Complete (Quotes Fixed) ---")
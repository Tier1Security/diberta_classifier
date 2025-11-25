import csv
import random
import os
import string

# --- CONFIGURATION ---
DATA_DIR = "data_4class_security"
TRAIN_FILE = os.path.join(DATA_DIR, "train.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation.csv")
TEST_FILE = os.path.join(DATA_DIR, "test.csv")

# Total samples (Balanced 4 ways)
TOTAL_EXAMPLES = 60000 
TRAIN_COUNT = int(TOTAL_EXAMPLES * 0.8)
VALIDATION_COUNT = int(TOTAL_EXAMPLES * 0.1)
TEST_COUNT = int(TOTAL_EXAMPLES * 0.1)

# --- NOISE ENGINES ---
def random_case(text):
    """Randomly capitalizes characters."""
    if random.random() < 0.3:
        return "".join(c.upper() if random.random() < 0.5 else c.lower() for c in text)
    return text.lower()

def random_space():
    return ' ' * random.randint(1, 3)

def random_filename(extension):
    name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
    return f"{name}.{extension}"

def maybe_quote(text):
    return f'"{text}"' if random.random() < 0.5 else text

# ==========================================
# 1. MALICIOUS REGISTRY (T1003)
# ==========================================
SAM_COMPONENTS = {
    "executables": ["reg.exe", "reg", "c:\\windows\\system32\\reg.exe"],
    "actions": ["save", "export"],
    "hives": ["hklm\\sam", "hklm\\system", "hklm\\security", "HKEY_LOCAL_MACHINE\\SAM"],
    "wrappers": ["", "cmd.exe /c {}", "powershell -c {}", "cmd /k \"{}\""]
}

# ==========================================
# 2. MALICIOUS FIREWALL (T1562)
# ==========================================
FW_MALICIOUS_COMPONENTS = {
    "names": ["Allow All", "WinUpdate Helper", "Netcat Listener", "Sliver C2", "Metasploit", "System Integrity", "Chrome Helper", "DNS Query", "PS Remoting"],
    "apps": ["nc.exe", "updater.exe", "powershell.exe", "cmd.exe", "scvhost.exe", "lsasss.exe", "svchost.exe"],
    "paths": ["C:\\Windows\\Temp", "C:\\Users\\Public", "C:\\Users\\Admin\\AppData\\Local\\Temp"],
    "directions": ["Inbound", "Outbound"],
    "actions": ["Allow"], 
    "ports": ["4444", "1337", "8888", "443", "80", "53", "8080", "Any", "*"]
}

# ==========================================
# 3. TOKEN MANIPULATION (T1134)
# ==========================================
TOKEN_COMPONENTS = {
    "privileges": ["SeDebugPrivilege", "SeBackupPrivilege", "SeRestorePrivilege", "SeTakeOwnershipPrivilege"],
    "api_calls": ["AdjustTokenPrivileges", "OpenProcessToken", "LookupPrivilegeValue", "NtQueueApcThread"],
    "snippets": ["Add-Type -MemberDefinition", "[DllImport(\"advapi32.dll\")]"]
}
REFLECTION_ATTACKS = [
    "[Ref].Assembly.GetType('System.Management.Automation.AmsiUtils')",
    "$a = [System.Runtime.InteropServices.Marshal]::GetDelegateForFunctionPointer",
    "System.Reflection.Assembly::Load($base64)"
]

# ==========================================
# 0. BENIGN COMPONENTS (Restored & Expanded)
# ==========================================
BENIGN_REG_SAVES = {
    "executables": ["reg.exe", "reg"],
    "actions": ["save", "export"], 
    "hives": ["hkcu\\Software", "HKLM\\Software\\MyCorp", "HKCU\\Console"],
    "folders": ["C:\\Backups", "C:\\Temp", "D:\\Logs"]
}
BENIGN_FW_SAFE = {
    "names": ["Core Networking", "Microsoft Teams", "Apache Web Server", "Node Server"],
    "apps": ["Teams.exe", "httpd.exe", "System", "chrome.exe", "node.exe"],
    "paths": ["C:\\Program Files\\Microsoft Teams", "C:\\Apache\\bin", "C:\\Program Files (x86)\\Google\\Chrome"],
    "directions": ["Outbound", "Inbound"], 
    "actions": ["Allow", "Block"],
    "ports": ["80", "443", "53", "3389", "8080", "3000"]
}
BENIGN_TOKEN_VERBS = ["Get-Command", "Get-Help", "Write-Host", "echo", "Select-String -Pattern"]
BENIGN_FILE_OPS = {
    "verbs": ["type", "cat", "Get-Content", "ls", "dir", "Copy-Item"],
    "scary_files": ["sam.save", "security.log", "SAM", "SYSTEM"]
}
BENIGN_LOG_SEARCH = {
    "verbs": ["Select-String", "findstr"],
    "targets": ["SeDebugPrivilege", "password", "error"],
    "files": ["C:\\Logs\\security.log", "C:\\Windows\\System32\\winevt\\Logs\\Security.evtx"]
}
BENIGN_REFLECTION = [
    "[System.Reflection.Assembly]::GetExecutingAssembly()",
    "[System.Math]::Pow(2, 10)",
    "$date = [System.DateTime]::Now"
]

# --- GENERATORS ---

def generate_sam_command():
    comp = SAM_COMPONENTS
    exe = random.choice(comp["executables"])
    action = random.choice(comp["actions"])
    hive = maybe_quote(random.choice(comp["hives"]))
    folder = random.choice(["C:\\Windows\\Temp", "C:\\Temp", "C:\\Users\\Public"])
    filename = random_filename("save")
    path = maybe_quote(f"{folder}\\{filename}")
    
    cmd = f"{exe}{random_space()}{action}{random_space()}{hive}{random_space()}{path}"
    wrapper = random.choice(comp['wrappers'])
    if wrapper: cmd = wrapper.format(cmd)
    return random_case(cmd), 1

def generate_firewall_malicious():
    comp = FW_MALICIOUS_COMPONENTS
    
    # FIX 1: "Allow All / Any Port" logic
    if random.random() < 0.15:
        return "rule: allow all | app: any | dir: inbound | act: allow | port: any", 2
        
    # FIX 2: "Wolf in Sheep's Clothing" (Bad App on Safe Port)
    # Force apps like nc.exe/powershell to use ports 53, 80, 443, 8080
    if random.random() < 0.20:
        rule = random.choice(["DNS Query", "HTTP Access", "System Update"])
        app_name = random.choice(["nc.exe", "powershell.exe", "cmd.exe"]) 
        app_path = random.choice(comp["paths"])
        app = f"{app_path}\\{app_name}"
        port = random.choice(["53", "80", "443", "8080"])
    else:
        # Standard Random Malicious
        rule = random.choice(comp["names"]) + " " + str(random.randint(1, 999))
        app_name = random.choice(comp["apps"])
        app_path = random.choice(comp["paths"])
        app = f"{app_path}\\{app_name}"
        port = random.choice(comp["ports"])

    feat = f"rule: {rule} | app: {app} | dir: {random.choice(comp['directions'])} | act: Allow | port: {port}"
    return feat.lower(), 2 

def generate_token_manipulation():
    comp = TOKEN_COMPONENTS
    rand = random.random()
    if rand < 0.4: 
        snippet = random.choice(comp['snippets'])
        cmd = f"{snippet} public static extern bool {random.choice(comp['api_calls'])}(...)"
    elif rand < 0.7: 
        cmd = f"SetTokenPriv::EnablePrivilege({random.choice(comp['privileges'])})"
    else: 
        base_cmd = random.choice(REFLECTION_ATTACKS)
        cmd = f"{base_cmd}; $val = {random.randint(0,9999)}"
    return random_case(cmd), 3

def generate_benign_mixed():
    """Generates Label 0 (Benign) with properly re-integrated edge cases."""
    rand = random.random()
    
    if rand < 0.20: # 20% Reg Saves
        comp = BENIGN_REG_SAVES
        filename = random_filename("reg")
        path = maybe_quote(f"{random.choice(comp['folders'])}\\{filename}")
        cmd = f"{random.choice(comp['executables'])} {random.choice(comp['actions'])} {maybe_quote(random.choice(comp['hives']))} {path}"
        return random_case(cmd), 0
        
    elif rand < 0.40: # 20% Safe Firewall
        comp = BENIGN_FW_SAFE
        app = f"{random.choice(comp['paths'])}\\{random.choice(comp['apps'])}"
        feat = f"rule: {random.choice(comp['names'])} | app: {app} | dir: {random.choice(comp['directions'])} | act: Allow | port: {random.choice(comp['ports'])}"
        return feat.lower(), 0
        
    elif rand < 0.55: # 15% Safe Token Checks (whoami /priv, Get-Help)
        cmd = f"{random.choice(BENIGN_TOKEN_VERBS)} {random.choice(TOKEN_COMPONENTS['privileges'])}"
        return random_case(cmd), 0
        
    elif rand < 0.70: # 15% Safe File Ops (FIX FOR 'type sam.save')
        comp = BENIGN_FILE_OPS
        # Read a scary file from a temp path
        path = f"C:\\Temp\\{random.choice(comp['scary_files'])}"
        cmd = f"{random.choice(comp['verbs'])} {maybe_quote(path)}"
        return random_case(cmd), 0
        
    elif rand < 0.85: # 15% Log Search (FIX FOR 'Select-String')
        comp = BENIGN_LOG_SEARCH
        cmd = f"{random.choice(comp['verbs'])} {random.choice(comp['targets'])} {random.choice(comp['files'])}"
        return random_case(cmd), 0
        
    else: # 15% Benign Reflection
        cmd = random.choice(BENIGN_REFLECTION)
        return random_case(cmd), 0

# --- MAIN LOOP ---

def create_dataset(file_path, target_per_class):
    print(f"Generating {file_path} (Target: {target_per_class} per class)...")
    output_dir = os.path.dirname(file_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    
    data = []
    generators = {
        0: generate_benign_mixed,
        1: generate_sam_command,
        2: generate_firewall_malicious,
        3: generate_token_manipulation
    }
    
    for label, generator_func in generators.items():
        unique_samples = set()
        attempts = 0
        max_attempts = target_per_class * 50 
        
        while len(unique_samples) < target_per_class and attempts < max_attempts:
            sample_text, sample_label = generator_func()
            unique_samples.add((sample_text, sample_label))
            attempts += 1
            
        if len(unique_samples) < target_per_class:
            print(f"[WARN] Only generated {len(unique_samples)} unique samples for Class {label}")
            
        data.extend(list(unique_samples))

    random.shuffle(data)
    
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"]) 
        writer.writerows(data)
    print(f"Created {file_path} with {len(data)} balanced samples.")

if __name__ == "__main__":
    create_dataset(TRAIN_FILE, TRAIN_COUNT // 4)
    create_dataset(VALIDATION_FILE, VALIDATION_COUNT // 4)
    create_dataset(TEST_FILE, TEST_COUNT // 4)
    print("\n--- 4-CLASS INFINITE GENERATION V7 COMPLETE ---")
import csv
import random
import os
import string

# --- CONFIGURATION ---
OUTPUT_FILE = "synthetic_benign_baseline.csv"
TARGET_COUNT = 50000 

# --- DATA LIBRARIES ---
USERNAMES = ["j.doe", "admin_svc", "hr_manager", "dev_01", "marketing_user", "system_local", "it_admin"]

BINARIES = {
    "OFFICE_APPS": ["winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe", "onenote.exe"],
    "BROWSERS": ["msedge.exe", "chrome.exe", "browser_assistant.exe", "brave.exe"],
    "DEV_TOOLS": ["git.exe", "npm.cmd", "node.exe", "python.exe", "pip.exe", "code.exe", "docker.exe"],
    "SYS_ADMIN": ["ipconfig.exe", "netstat.exe", "tasklist.exe", "systeminfo.exe", "whoami.exe", "hostname.exe", "gpupdate.exe", "sc.exe", "net.exe", "tzutil.exe", "chcp.com"],
    "NET_TOOLS": ["ping.exe", "nslookup.exe", "tracert.exe", "curl.exe", "ssh.exe"],
    "POWERSHELL": ["powershell.exe", "pwsh.exe"],
    "OS_SERVICES": ["svchost.exe", "runtimebroker.exe", "searchindexer.exe", "backgroundtaskhost.exe", "lsass.exe"]
}

FLAGS = {
    "OFFICE": ["/n", "/e", "/recycle", "/s", "/pt", "/noui", "/background", "/embedding"],
    "BROWSER": [
        "--type=renderer", "--no-sandbox", "--process-per-site", 
        "--service-sandbox-type=none", "--lang=en-US", "--enable-logging",
        "--mojo-platform-channel-handle", "--background-update", "--no-startup-window"
    ],
    "PS_BENIGN": [
        "-NoProfile", "-ExecutionPolicy Bypass", "-WindowStyle Hidden", 
        "-Command \"Get-Content\"", "-Command \"Get-Date\"", "-NonInteractive",
        "-Command \"Get-Service\"", "-Command \"Get-Process\""
    ],
    "DEV": ["install", "fetch", "pull", "status", "build", "run", "--version", "-m", "list", "ps -a"],
    "ADMIN": ["/all", "/v", "/s", "/fo list", "/nh", "/groups", "/priv", "/force"],
    
    # --- HARDENING COUNTER-WEIGHTS ---
    "SERVICE_HARDENING": ["query", "queryex", "enumdepend", "getdisplayname", "qdescription", "qc"],
    "PROCESS_HARDENING": ["/v", "/svc", "/fi \"status eq running\"", "/fo table", "/m"],
    "LOCATION_HARDENING": ["/g", "/s localhost", "/fo csv"],
    "NET_BENIGN": ["start", "view", "use", "session", "statistics server"],
    
    "NETWORK": ["-n 1", "-t", "-a", "-L", "https://api.internal.local", "-V"]
}

PATH_ROOTS = [
    "C:\\Users\\{user}\\",
    "C:\\Program Files\\",
    "C:\\Windows\\System32\\",
    "C:\\ProgramData\\",
    "%TEMP%\\",
    "%APPDATA%\\Local\\",
    "%USERPROFILE%\\Documents\\"
]

SUB_DIRS = ["AppData\\Local", "Documents\\Work", "Temp\\Cache", "Bin\\Win64", "Roaming\\Microsoft", "Downloads\\Legacy"]
EXTENSIONS = [".docx", ".xlsx", ".pptx", ".js", ".py", ".log", ".tmp", ".dll", ".json", ".ini"]

def normalize_command(cmd):
    """
    Identical to Agent Normalization: Ensures training data matches inference data.
    """
    if not isinstance(cmd, str): return ""
    cmd = cmd.lower()
    cmd = cmd.replace("^", "").replace("`", "")
    cmd = cmd.replace("/", "\\")
    cmd = " ".join(cmd.split())
    cmd = cmd.replace("> nul 2>&1", "").replace(">nul", "")
    cmd = cmd.replace("\\.\\", "\\")
    return cmd.strip()

def get_random_hex(length=8):
    return ''.join(random.choice(string.hexdigits) for _ in range(length)).lower()

def generate_heuristic_command():
    category_choices = list(BINARIES.keys())
    weights = [1.0, 1.0, 1.0, 3.0, 1.0, 1.0, 1.0] 
    category = random.choices(category_choices, weights=weights, k=1)[0]
    
    bin_name = random.choice(BINARIES[category])
    user = random.choice(USERNAMES)
    
    root = random.choice(PATH_ROOTS).format(user=user)
    sub = random.choice(SUB_DIRS)
    ext = random.choice(EXTENSIONS)
    file_target = f"data_{random.randint(1000, 9999)}{ext}"
    full_path = f"{root}{sub}\\{file_target}"
    
    q = random.choice(['"', "'", ""])

    if category == "OFFICE_APPS":
        flag = random.choice(FLAGS["OFFICE"])
        return f"{bin_name} {flag} {q}{full_path}{q}"
    
    elif category == "BROWSERS":
        f_count = random.randint(2, 5)
        selected_flags = " ".join(random.sample(FLAGS["BROWSER"], f_count))
        url = f"https://{random.choice(['portal', 'wiki', 'hr', 'jira'])}-{random.randint(1,9)}.internal"
        return f"{bin_name} {selected_flags} --url {url}"
    
    elif category == "POWERSHELL":
        flag = random.choice(FLAGS["PS_BENIGN"])
        return f"{bin_name} {flag}"

    elif category == "DEV_TOOLS":
        verb = random.choice(FLAGS["DEV"])
        if bin_name in ["npm.cmd", "pip.exe"]:
            return f"{bin_name} {verb} {random.choice(['requests', 'pandas', 'express', 'react'])} --quiet"
        return f"{bin_name} {verb} {q}{full_path}{q}"
    
    elif category == "SYS_ADMIN":
        if bin_name == "sc.exe":
            flag = random.choice(FLAGS["SERVICE_HARDENING"])
            return f"{bin_name} {flag}"
        elif bin_name == "tasklist.exe":
            flag = random.choice(FLAGS["PROCESS_HARDENING"])
            return f"{bin_name} {flag}"
        elif bin_name == "net.exe":
            flag = random.choice(FLAGS["NET_BENIGN"])
            return f"{bin_name} {flag}"
        elif bin_name in ["tzutil.exe", "systeminfo.exe"]:
            flag = random.choice(FLAGS["LOCATION_HARDENING"])
            return f"{bin_name} {flag}"
        else:
            flag = random.choice(FLAGS["ADMIN"])
            return f"{bin_name} {flag}"
    
    elif category == "NET_TOOLS":
        flag = random.choice(FLAGS["NETWORK"])
        return f"{bin_name} {flag}"
    
    else: # OS_SERVICES
        svcs = ['netsvcs', 'localservice', 'networkservice', 'apphost']
        return f"{bin_name} -k {random.choice(svcs)} -p {get_random_hex(4)}"

def build_dataset(count):
    print(f"[*] Synthesizing {count} hardened and NORMALIZED heuristic commands...")
    print(f"[*] Hardening Focus: T1007, T1057, T1614.")
    commands = set()
    
    while len(commands) < count:
        raw_cmd = generate_heuristic_command()
        
        # Apply normalization immediately to mirror production logs
        clean_cmd = normalize_command(raw_cmd)
        
        if clean_cmd:
            commands.add(clean_cmd)
        
        if len(commands) % 100000 == 0:
            print(f"[*] Progress: {len(commands)} commands generated...")

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["command", "label"])
        for cmd in commands:
            writer.writerow([cmd, 0])
            
    print(f"[+] Baseline generation complete: {OUTPUT_FILE}")

if __name__ == "__main__":
    build_dataset(TARGET_COUNT)
import csv
import random
import os

# --- CONFIGURATION ---
OUTPUT_FILE = "synthetic_benign_baseline.csv"
TARGET_COUNT = 500000

# --- HEURISTIC DATA LIBRARIES ---
# Grouped by "Behavioral Profile"
BINARIES = {
    "OFFICE_APPS": ["winword.exe", "excel.exe", "powerpnt.exe", "outlook.exe", "onenote.exe"],
    "BROWSERS": ["msedge.exe", "chrome.exe", "browser_assistant.exe"],
    "DEV_TOOLS": ["git.exe", "npm.cmd", "node.exe", "python.exe", "pip.exe", "code.exe"],
    "SYS_ADMIN": ["ipconfig.exe", "netstat.exe", "tasklist.exe", "systeminfo.exe", "whoami.exe", "hostname.exe"],
    "NET_TOOLS": ["ping.exe", "nslookup.exe", "tracert.exe", "curl.exe", "ssh.exe"],
    "OS_SERVICES": ["svchost.exe", "runtimebroker.exe", "searchindexer.exe", "backgroundtaskhost.exe"]
}

# Heuristic Flag sets - Focus on the "Grammar" of the binary
FLAGS = {
    "OFFICE": ["/n", "/e", "/recycle", "/s", "/pt", "/noui", "/background", "/embedding"],
    "BROWSER": [
        "--type=renderer", "--no-sandbox", "--process-per-site",
        "--service-sandbox-type=none", "--lang=en-US", "--enable-logging",
        "--mojo-platform-channel-handle", "--background-update"
    ],
    "DEV": ["install", "fetch", "pull", "status", "build", "run", "--version", "-m", "list"],
    "ADMIN": ["/all", "/v", "/s", "/fo list", "/nh", "/groups", "/priv"],
    "NETWORK": ["-n 1", "-t", "-a", "-L", "https://internal-api.corp.local", "-V"]
}

# Generic path structures to prevent overfitting on specific folder names
PATH_ROOTS = [
    "C:\\Users\\<USER>\\",
    "C:\\Program Files\\",
    "C:\\Windows\\System32\\",
    "\\\\RemoteShare\\Data\\",
    "D:\\Project_Storage\\"
]

# Random "Noise" directories to ensure path entropy
SUB_DIRS = ["AppData\\Local", "Documents", "Temp", "Bin", "Roaming\\App", "Downloads\\Cache"]

EXTENSIONS = [".docx", ".xlsx", ".pptx", ".js", ".py", ".log", ".tmp", ".dll", ".json"]

def generate_heuristic_command():
    category = random.choice(list(BINARIES.keys()))
    bin_name = random.choice(BINARIES[category])

    # Construct a "Behavioral Path"
    root = random.choice(PATH_ROOTS)
    sub = random.choice(SUB_DIRS)
    ext = random.choice(EXTENSIONS)
    file_target = f"file_{random.randint(100, 999)}{ext}"
    full_path = f"{root}{sub}\\{file_target}"

    # Logic-based Flag Mixing
    if category == "OFFICE_APPS":
        flag = random.choice(FLAGS["OFFICE"])
        return f"{bin_name} {flag} \"{full_path}\""

    elif category == "BROWSERS":
        # Browsers often use multiple complex flags
        f_count = random.randint(2, 4)
        selected_flags = " ".join(random.sample(FLAGS["BROWSER"], f_count))
        # Add a random internal URL heuristic
        url = f"https://{random.choice(['portal', 'sharepoint', 'wiki'])}-{random.randint(1,5)}.local"
        return f"{bin_name} {selected_flags} {url}"

    elif category == "DEV_TOOLS":
        verb = random.choice(FLAGS["DEV"])
        return f"{bin_name} {verb} \"{full_path}\""

    elif category == "SYS_ADMIN":
        flag = random.choice(FLAGS["ADMIN"])
        return f"{bin_name} {flag}"

    elif category == "NET_TOOLS":
        flag = random.choice(FLAGS["NETWORK"])
        return f"{bin_name} {flag}"

    else: # OS_SERVICES
        # System services often have "K" switches or hex identifiers
        return f"{bin_name} -k {random.choice(['netsvcs', 'localservice', 'networkservice'])} -p"

def build_dataset(count):
    print(f"[*] Synthesizing {count} heuristic-focused benign commands...")
    commands = set()

    while len(commands) < count:
        commands.add(generate_heuristic_command())
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

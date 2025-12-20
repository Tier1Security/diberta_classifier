import subprocess
import time
import random
import os

# ==========================================
# BENIGN 1: PURE NOISE (Daily Activity)
# Now expanded with MS Office & Productivity tools
# ==========================================
Benign_Pure_Noise = [
    # Core Productivity (MS Office)
    "winword.exe /n", # Word new instance
    "excel.exe /e",   # Excel embed mode
    "outlook.exe /recycle",
    "powerpnt.exe /s", # PowerPoint splash/slideshow
    "onenote.exe /quicknote",

    # Browsers & Communication
    "msedge.exe --no-startup-window",
    "chrome.exe --type=utility --utility-sub-type=network.mojom.NetworkService",
    "teams.exe --process-per-site",
    "slack.exe --startup",

    # Standard OS Noise
    "explorer.exe",
    "tasklist /v",
    "systeminfo",
    "whoami /groups",
    "ipconfig /all",
    "ping 127.0.0.1 -n 1",
    "dir c:\\users",
    "notepad.exe",
    "control.exe /name microsoft.windowsupdate"
]

# ==========================================
# BENIGN 2: HARD NEGATIVES (Enterprise IT & Infrastructure)
# Expanded with Update Services, Sync, and Management Tools
# ==========================================
Benign_Hard_Negatives = [
    # Cloud Sync & Background Updates (High Variance Syntax)
    "onedrive.exe /background",
    "dropbox.exe /home",
    "googleupdate.exe /c",
    "adobearm.exe /nosplash", # Adobe Reader Update

    # IT Administration & Telemetry (Keyword-Heavy but Safe)
    "wmic product get name, version",
    "get-help adjusttokenprivileges -full",
    "sc query windefend",
    "netsh advfirewall show allprofiles",
    "gpresult /r /scope user",
    "wevtutil qe security /f:text /c:1", # Read last security event

    # System Maintenance (The "Scary" but Benign)
    "netsh dump > netconfig.txt",
    "backup-utility save registry config",
    "reg query hkcu\\software\\microsoft\\windows\\currentversion\\run",
    "cleanmgr.exe /sagerun:1",

    # Developer/Admin Processes
    "python -m http.server 8080",
    "node server.js --port 3000",
    "git fetch --all",
    "docker stats --no-stream"
]

def run_emulation(profile="balanced"):
    """
    Simulates user noise by executing benign commands.
    Ensures the Anomaly Engine learns the 'Toxin-Free' baseline.
    """
    print(f"[*] Starting {profile} noise emulation for Security Agent X...")
    print(f"[*] Inclusion: MS Office, Cloud Sync, and IT Management noise active.")

    if profile == "pure_noise":
        delay_range = (5, 15)
    elif profile == "hard_negatives":
        delay_range = (45, 120)
    else: # balanced
        delay_range = (10, 45)

    try:
        while True:
            rng = random.random()

            if profile == "pure_noise":
                cmd = random.choice(Benign_Pure_Noise)
            elif profile == "hard_negatives":
                cmd = random.choice(Benign_Hard_Negatives)
            else:
                # 75% Pure Noise (Office/Browser), 25% Hard Negatives (IT/Updates)
                if rng < 0.75:
                    cmd = random.choice(Benign_Pure_Noise)
                else:
                    cmd = random.choice(Benign_Hard_Negatives)

            print(f"[>] Simulating: {cmd}")

            try:
                # We use shell=True logic to ensure Event ID 4688 is triggered correctly
                # Timeout is short because we only need the process creation log entry
                subprocess.run(["powershell", "-Command", cmd],
                               capture_output=True,
                               timeout=5)
            except Exception:
                pass

            wait_time = random.randint(*delay_range)
            print(f"[*] Next event in {wait_time}s...")
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("\n[!] Emulation stopped. Baseline collection complete.")

if __name__ == "__main__":
    # RECOMMENDATION: Run 'balanced' to get the best mix of productivity and system noise.
    run_emulation("balanced")

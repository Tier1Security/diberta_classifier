import torch
from transformers import pipeline
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURATION ---
MODEL_PATH = "models/merged_4class_roberta"

# --- THE "RED TEAM" DATASET ---
# Hand-crafted examples that differ significantly from the training generator
test_cases = [
    # ============================================
    # 1. T1003 (REGISTRY DUMPING) - Real World Variations
    # ============================================
    # Standard Attack
    {"text": "reg save HKLM\\SAM C:\\Windows\\Temp\\sam.hiv", "label": "T1003.002", "desc": "Standard SAM Save"},
    # Weird Casing (Obfuscation)
    {"text": "rEg.eXe sAvE hklM\\sEcUrItY c:\\temp\\sec.dump", "label": "T1003.002", "desc": "Mixed Case Obfuscation"},
    # Using 'export' instead of 'save' on Sensitive Hive
    {"text": "reg export HKLM\\SYSTEM C:\\Users\\Public\\system.reg /y", "label": "T1003.002", "desc": "Exporting SYSTEM Hive"},
    # Short Hive Name
    {"text": "cmd /c reg save HKLM\\SAM sam.save", "label": "T1003.002", "desc": "Short path in CMD wrapper"},
    # PowerShell Wrapper
    {"text": "powershell -nop -c \"reg save HKLM\\Security security.hive\"", "label": "T1003.002", "desc": "PowerShell Wrapper"},
    
    # ============================================
    # 2. T1562 (FIREWALL) - Mimicry & Evasion
    # ============================================
    # Malware hiding in AppData
    {"text": "rule: Chrome Helper | app: C:\\Users\\Admin\\AppData\\Local\\Temp\\chrome.exe | dir: Outbound | act: Allow | port: 443", "label": "T1562", "desc": "Malware in AppData Temp"},
    # Netcat pretending to be DNS
    {"text": "rule: DNS Query | app: C:\\Windows\\Temp\\nc.exe | dir: Inbound | act: Allow | port: 53", "label": "T1562", "desc": "Netcat pretending to be DNS"},
    # Typosquatting (scvhost vs svchost)
    {"text": "rule: System Service | app: C:\\Windows\\System32\\scvhost.exe | dir: Inbound | act: Allow | port: 4444", "label": "T1562", "desc": "Typosquatting scvhost.exe"},
    # Opening "Any" port for "Any" app (The "Lazy Admin" / Backdoor)
    {"text": "rule: Allow All | app: Any | dir: Inbound | act: Allow | port: Any", "label": "T1562", "desc": "Allow All / Any Port"},
    # Powershell opening listener
    {"text": "rule: PS Remoting | app: powershell.exe | dir: Inbound | act: Allow | port: 8080", "label": "T1562", "desc": "PowerShell Listener on 8080"},

    # ============================================
    # 3. T1134 (TOKEN MANIPULATION) - Code Variations
    # ============================================
    # Standard P/Invoke
    {"text": "[DllImport(\"advapi32.dll\")] public static extern bool AdjustTokenPrivileges(IntPtr TokenHandle, bool DisableAllPrivileges, ref TOKEN_PRIVILEGES NewState, int BufferLength, IntPtr PreviousState, IntPtr ReturnLength);", "label": "T1134", "desc": "Full C# Signature"},
    # Shortened / slightly different variable names
    {"text": "[DllImport(\"advapi32.dll\")] extern bool AdjustTokenPrivileges(IntPtr hTok, bool disAll, ref TokPriv NewSt, int len, IntPtr prev, IntPtr retLen)", "label": "T1134", "desc": "Varied Variable Names"},
    # Using 'OpenProcessToken' in a suspicious context
    {"text": "$handle = [System.Diagnostics.Process]::GetCurrentProcess().Handle; OpenProcessToken($handle, ...)", "label": "T1134", "desc": "OpenProcessToken Call"},
    # Enabling SeBackupPrivilege via function
    {"text": "SetTokenPriv::EnablePrivilege(SeBackupPrivilege)", "label": "T1134", "desc": "Enable SeBackupPrivilege"},
    # Reflection based load
    {"text": "$a = [Ref].Assembly.GetType('System.Management.Automation.AmsiUtils')", "label": "T1134", "desc": "Reflection (General Suspicious)"}, # This might fail if untrained, but good to test

    # ============================================
    # 0. BENIGN - The "Trap" Questions
    # ============================================
    # Safe Registry Export (Looks like T1003 but safe hive)
    {"text": "reg export HKLM\\Software\\Policies policy_backup.reg", "label": "Benign", "desc": "Exporting Safe Hive (Policies)"},
    # Safe Registry Save (Console settings)
    {"text": "reg save HKCU\\Console console.config", "label": "Benign", "desc": "Saving Console Settings"},
    # Reading a file named 'sam.save' (Not executing it)
    {"text": "type C:\\Temp\\sam.save", "label": "Benign", "desc": "Reading a file named 'sam.save'"},
    # Python Web Server (Dev activity)
    {"text": "rule: Python Dev Server | app: C:\\Python39\\python.exe | dir: Inbound | act: Allow | port: 8000", "label": "Benign", "desc": "Python Dev Server Inbound"},
    # Node.js Server
    {"text": "rule: Node Backend | app: C:\\Program Files\\nodejs\\node.exe | dir: Inbound | act: Allow | port: 3000", "label": "Benign", "desc": "NodeJS Server Inbound"},
    # Admin checking privs
    {"text": "whoami /priv", "label": "Benign", "desc": "Admin checking privs"},
    # Admin searching for help on the scary command
    {"text": "Get-Help AdjustTokenPrivileges -Full", "label": "Benign", "desc": "Help Lookup for Token Cmd"},
    # Searching logs for an attack
    {"text": "Select-String -Path C:\\Logs\\security.log -Pattern 'SeDebugPrivilege'", "label": "Benign", "desc": "Searching logs for Privilege"},
    # Common Admin Command that looks weird
    {"text": "wmic process get commandline, processid", "label": "Benign", "desc": "WMIC Process List"}
]

def main():
    print(f"--- Loading Model: {MODEL_PATH} ---")
    try:
        classifier = pipeline(
            "text-classification", 
            model=MODEL_PATH, 
            tokenizer=MODEL_PATH, 
            device=0 if torch.cuda.is_available() else -1
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"--- Running {len(test_cases)} Real-World Scenarios ---")
    print(f"{'Description':<35} | {'True Label':<12} | {'Predicted':<12} | {'Conf':<6} | {'Result'}")
    print("-" * 85)

    y_true = []
    y_pred = []
    failures = 0

    for case in test_cases:
        text = case["text"].lower() # Ensure input is lowercased like training
        true_label = case["label"]
        
        # Inference
        result = classifier(text)[0]
        pred_label = result['label']
        score = result['score']
        
        y_true.append(true_label)
        y_pred.append(pred_label)

        # Output
        is_correct = (pred_label == true_label)
        status = "✅" if is_correct else "❌"
        if not is_correct: failures += 1
        
        print(f"{case['desc']:<35} | {true_label:<12} | {pred_label:<12} | {score:.2f}   | {status}")

    # Summary
    accuracy = accuracy_score(y_true, y_pred)
    print("\n" + "="*40)
    print(f"REAL WORLD ACCURACY: {accuracy:.2%} ({len(test_cases)-failures}/{len(test_cases)})")
    print("="*40)
    
    if failures > 0:
        print("\n[!] Analysis: The model struggled with the inputs marked ❌.")
        print("    If accuracy is > 90%, the model is robust.")
        print("    If accuracy is < 80%, consider adding these specific failures to training data.")
    else:
        print("\n[+] The model survived the Red Team stress test!")

if __name__ == "__main__":
    main()
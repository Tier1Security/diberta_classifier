import torch
import time
import random
import csv
import datetime
import os
from transformers import pipeline
import re 

# --- CONFIGURATION ---
MODEL_PATH = "models/merged_4class_roberta"
ITERATIONS = 50 
CONFIDENCE_THRESHOLD = 0.90 
SEED = 42 
LOG_FAILURES = True 

# --- THE "RED TEAM" SCENARIOS V7 (Maximum Hardness) ---
base_scenarios = [
    # --- T1003 (REGISTRY/LSASS) ---
    {"text": "reg save hklm\\sam c:\\windows\\temp\\sam.hiv", "label": "T1003.002", "desc": "Standard SAM Save"},
    {"text": "reg export hklm\\system c:\\users\\public\\system.reg /y", "label": "T1003.002", "desc": "Exporting SYSTEM Hive (Malicious)"},
    {"text": "powershell -nop -c \"reg save hklm\\security security.hive\"", "label": "T1003.002", "desc": "PowerShell Wrapped Security Dump"},
    {"text": "rundll32.exe c:\\windows\\system32\\comsvcs.dll, minidump 1234 c:\\temp\\lsass.dmp full", "label": "T1003.002", "desc": "LSASS Minidump"},
    {"text": "type c:\\temp\\sam.save", "label": "T1003.002", "desc": "Reading a 'Scary' File"}, 
    
    # --- T1562 (FIREWALL/DEFENSE) ---
    {"text": "set-mppreference -disablerealtimemonitoring $true", "label": "T1562", "desc": "Disable Realtime Monitoring"},
    {"text": "sc stop windefend", "label": "T1562", "desc": "Stop Defender Service"},
    {"text": "rule: chrome helper | app: c:\\users\\admin\\appdata\\local\\temp\\chrome.exe | dir: outbound | act: allow | port: 443", "label": "T1562", "desc": "Malware in AppData (Malicious Log)"}, 
    {"text": "rule: dns query | app: c:\\windows\\temp\\nc.exe | dir: inbound | act: allow | port: 53", "label": "T1562", "desc": "Netcat DNS Impersonation (Malicious Log)"},

    # --- T1134 (TOKEN) ---
    {"text": "[dllimport(\"advapi32.dll\")] public static extern bool adjusttokenprivileges(...)", "label": "T1134", "desc": "Full C# AdjustToken Signature"},
    {"text": "$handle = [system.diagnostics.process]::getcurrentprocess().handle; openprocesstoken($handle, ...)", "label": "T1134", "desc": "OpenProcessToken Call"},
    {"text": "settokenpriv::enableprivilege(sebackupprivilege)", "label": "T1134", "desc": "Enable SeBackupPrivilege"},
    {"text": "$a = [ref].assembly.gettype('system.management.automation.amsiutils')", "label": "T1134", "desc": "Reflection (AMSI Bypass)"}, 

    # --- BENIGN HARD NEGATIVES (Admin Ambiguity) ---
    {"text": "reg query hklm\\software\\policies", "label": "Benign", "desc": "Exporting Safe Hive (Reg Query)"},
    {"text": "wmic product get name, version", "label": "Benign", "desc": "WMIC Product List (Safe/Product)"},
    {"text": "sc query windefend", "label": "Benign", "desc": "SC Query (Safe Service Check)"},
    {"text": "get-help adjusttokenprivileges -full", "label": "Benign", "desc": "Help Lookup (T1134 Hard Negative)"},
    {"text": "whoami /groups", "label": "Benign", "desc": "Admin checking groups (Safe)"},
    {"text": "c:\\windows\\system32\\svchost.exe -k localservicenonetworkfirewall -p", "label": "Benign", "desc": "OS Process Startup (Svchost)"},

    # --- BENIGN PURE NOISE (System Stability) ---
    {"text": "ipconfig /all", "label": "Benign", "desc": "System Recon (Network/IP)"},
    {"text": "notepad.exe", "label": "Benign", "desc": "System Noise (Simple App)"},
    {"text": "tasklist /v", "label": "Benign", "desc": "System Noise (Process List)"},
    {"text": "python -m http.server 8000", "label": "Benign", "desc": "Python Dev Server (CLI Format)"}, 
    {"text": "select-string -path c:\\logs\\security.log -pattern 'failed logon'", "label": "Benign", "desc": "Searching Logs (Safe Pattern)"}
]
def main():
    if SEED is not None:
        print(f"--- Running Deterministic Mode (Seed: {SEED}) ---")
        random.seed(SEED)
        torch.manual_seed(SEED)
    else:
        print("--- Running Standard Test Mode ---")

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

    # For metrics: track durations and peak VRAM usage (if CUDA available)
    durations = []
    peak_memory_bytes = 0
    using_cuda = torch.cuda.is_available()
    if using_cuda:
        try:
            torch.cuda.reset_peak_memory_stats(0)
        except Exception:
            # Older PyTorch may not support device arg
            torch.cuda.reset_peak_memory_stats()
    print(f"--- Starting {ITERATIONS} Iterations (Total {len(base_scenarios)*ITERATIONS} Tests) ---")
    
    failures = []
    low_confidence = []
    total_tests = 0
    
    # NEW: Sets to track unique failures to prevent repetition
    logged_failures_set = set()
    logged_low_confidence_set = set()

    for i in range(1, ITERATIONS + 1):
        print(f"\rRunning Iteration {i}/{ITERATIONS}...", end="")
        
        for case in base_scenarios:
            total_tests += 1
            
            # 1. Input Preparation (No Mutation)
            raw_text = case["text"]
            text_before_cleaning = raw_text 
            
            # 2. NORMALIZE (Enforce Lowercase and Single-Space Rule)
            cleaned_text = re.sub(r'\s+', ' ', text_before_cleaning.strip())
            final_input = cleaned_text.lower()
            
            true_label = case["label"]
            
            # Create a unique key for this input/label combination
            unique_key = (final_input, true_label)
            
            # Predict
            if using_cuda:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            start_time = time.perf_counter()
            result = classifier(final_input)[0]
            if using_cuda:
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            end_time = time.perf_counter()
            durations.append(end_time - start_time)

            if using_cuda:
                try:
                    cur_peak = torch.cuda.max_memory_allocated(0)
                except Exception:
                    cur_peak = torch.cuda.max_memory_allocated()
                if cur_peak > peak_memory_bytes:
                    peak_memory_bytes = cur_peak
            pred_label = result['label']
            score = result['score']
            
            # Logic: Check correctness
            is_correct = (pred_label == true_label)
            
            if not is_correct:
                if unique_key not in logged_failures_set: # Check for uniqueness
                    logged_failures_set.add(unique_key)
                    fail_data = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "input_raw": text_before_cleaning, 
                        "input_normalized": final_input,
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "confidence": score,
                        "desc": case['desc']
                    }
                    failures.append(fail_data)
                
            elif score < CONFIDENCE_THRESHOLD:
                if unique_key not in logged_low_confidence_set: # Check for uniqueness
                    logged_low_confidence_set.add(unique_key)
                    low_conf_data = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "input_raw": text_before_cleaning, 
                        "input_normalized": final_input,
                        "true_label": true_label,
                        "predicted_label": pred_label, 
                        "confidence": score,
                        "desc": case['desc']
                    }
                    low_confidence.append(low_conf_data)

    print("\n\n" + "="*80)
    print(f"STRESS TEST REPORT ({total_tests} Total Samples Processed)")
    print("="*80)

    # 1. REPORT FAILURES (DISPLAY)
    if failures:
        print(f"\n[!!!] FOUND {len(failures)} UNIQUE FAILURES (Incorrect Predictions):")
        print("-" * 80)
        print(f"{'Description':<25} | {'Predicted':<10} | {'Conf':<6} | {'Input Fragment'}")
        print("-" * 80)
        for f in failures:
            short_input = (f['input_normalized'][:40] + '...') if len(f['input_normalized']) > 40 else f['input_normalized']
            print(f"{f['desc']:<25} | {f['predicted_label']:<10} | {f['confidence']:.2f}   | {short_input}")
    else:
        print("\n[+] 0 Failures. The model was 100% accurate across all variations.")

    # 2. REPORT LOW CONFIDENCE (DISPLAY)
    if low_confidence:
        print(f"\n[?] FOUND {len(low_confidence)} UNIQUE LOW CONFIDENCE PASSES (Correct but < {CONFIDENCE_THRESHOLD}):")
        print("-" * 80)
        print(f"{'Description':<25} | {'True Label':<10} | {'Conf':<6} | {'Input Fragment'}")
        print("-" * 80)
        for l in low_confidence:
            short_input = (l['input_normalized'][:40] + '...') if len(l['input_normalized']) > 40 else l['input_normalized']
            print(f"{l['desc']:<25} | {l['true_label']:<10} | {l['confidence']:.2f}   | {short_input}")
    else:
        print("\n[+] 0 Low Confidence results. Model was > 90% sure on everything.")

    print("="*80)

    # 4. PERFORMANCE SUMMARY
    if durations:
        avg_time = sum(durations) / len(durations)
        median_time = sorted(durations)[len(durations)//2]
        print(f"\n[+] Average Time per Test: {avg_time:.6f} seconds ({avg_time*1000:.2f} ms)")
        print(f"[+] Median Time per Test: {median_time:.6f} seconds ({median_time*1000:.2f} ms)")
    else:
        print("\n[!] No duration data collected.")

    if using_cuda:
        peak_mb = peak_memory_bytes / (1024.0 ** 2)
        print(f"[+] Peak GPU Memory Used: {peak_mb:.2f} MB ({peak_memory_bytes} bytes)")
    else:
        print("[+] No CUDA device detected; VRAM metrics not applicable (CPU run).")

    # 3. SAVE FAILURES & LOW CONFIDENCE TO CSV
    if LOG_FAILURES and (failures or low_confidence):
        
        # Log Failures
        if failures:
            fail_filename = f"failures_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            print(f"\n[+] Saving {len(failures)} unique failures to {fail_filename}...")
            
            with open(fail_filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["timestamp", "input_raw", "input_normalized", "true_label", "predicted_label", "confidence", "desc"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(failures)
        
        # Log Low Confidence
        if low_confidence:
            lc_filename = f"low_confidence_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            print(f"\n[+] Saving {len(low_confidence)} unique low confidence passes to {lc_filename}...")
            
            with open(lc_filename, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ["timestamp", "input_raw", "input_normalized", "true_label", "predicted_label", "confidence", "desc"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(low_confidence)
        
if __name__ == "__main__":
    main()
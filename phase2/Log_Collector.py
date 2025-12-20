import win32evtlog
import win32evtlogutil
import win32security
import csv
import re

def get_process_events(log_type="Security", event_id=4688, max_events=100000):
    """
    Queries the Windows Event Log for Process Creation (4688) events.
    Extracts the 'Process Command Line' field.
    """
    print(f"[*] Querying {log_type} log for Event ID {event_id}...")

    server = 'localhost'
    handle = win32evtlog.OpenEventLog(server, log_type)
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ

    total_found = 0
    commands = set() # Use a set to automatically deduplicate

    while True:
        events = win32evtlog.ReadEventLog(handle, flags, 0)
        if not events:
            break

        for event in events:
            if event.EventID == event_id:
                # Event ID 4688 data is stored in StringInserts
                # Position 8 is usually the Process Command Line
                try:
                    inserts = event.StringInserts
                    if inserts and len(inserts) > 8:
                        cmd_line = inserts[8].strip()

                        # Basic Cleaning
                        if cmd_line and cmd_line != "-":
                            # Remove excessive quotes or whitespace
                            cmd_line = cmd_line.replace('"', '').lower()
                            commands.add(cmd_line)
                            total_found += 1
                except Exception:
                    continue

        if total_found >= max_events:
            break

    return list(commands)

def save_to_baseline(commands, filename="benign_baseline.csv"):
    """
    Saves the extracted commands to a CSV for training.
    """
    print(f"[*] Saving {len(commands)} unique commands to {filename}...")
    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["command", "label"]) # label 0 for benign
        for cmd in commands:
            writer.writerow([cmd, 0])
    print("[+] Baseline collection complete.")

if __name__ == "__main__":
    # Note: Ensure "Audit Process Creation" is enabled in Local Group Policy
    # and "Include command line in process creation events" is enabled.
    try:
        baseline_data = get_process_events()
        if baseline_data:
            save_to_baseline(baseline_data)
        else:
            print("[!] No events found. Ensure Audit Process Creation is enabled.")
    except Exception as e:
        print(f"[!] Error: {e}")
        print("[i] Hint: Run this script as Administrator.")

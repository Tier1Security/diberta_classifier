import os
import yaml
import csv
import re

def scrape_atomic_red_team(atomics_path):
    """
    Parses the Atomic Red Team folder to extract MITRE IDs and their associated commands.
    Expected path: 'atomic-red-team/atomics'
    """
    atlas_data = []

    if not os.path.exists(atomics_path):
        print(f"[!] Path not found: {atomics_path}")
        print("[i] Please clone the repo first: git clone https://github.com/redcanaryco/atomic-red-team.git")
        return None

    print(f"[*] Scouring {atomics_path} for MITRE techniques...")

    # Walk through the atomics directory (T1003, T1059, etc.)
    for root, dirs, files in os.walk(atomics_path):
        for file in files:
            if file.endswith(".yaml") and re.match(r"T\d{4}.*\.yaml", file):
                file_path = os.path.join(root, file)
                mitre_id = file.split('.')[0] # e.g., T1003

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)

                        if not data or 'atomic_tests' not in data:
                            continue

                        for test in data['atomic_tests']:
                            test_name = test.get('name', 'Unnamed Test')
                            executor = test.get('executor', {})

                            # We want the command that actually runs
                            command = executor.get('command', '')
                            if not command:
                                # Sometimes it's a multi-line script in 'powershell_command' or similar
                                command = executor.get('powershell_command', '')

                            if command:
                                # Clean the command (remove newlines, extra spaces)
                                clean_cmd = " ".join(command.split()).strip()

                                atlas_data.append({
                                    'mitre_id': mitre_id,
                                    'technique_name': test_name,
                                    'command': clean_cmd.lower()
                                })
                except Exception as e:
                    # Some YAML files might have custom tags that fail safe_load
                    continue

    print(f"[+] Extraction complete. Found {len(atlas_data)} malicious command templates.")
    return atlas_data

def save_to_atlas(data, filename="mitre_atlas_raw.csv"):
    """
    Saves the extracted MITRE commands to a CSV.
    """
    keys = data[0].keys()
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    print(f"[+] Malicious Atlas saved to {filename}")

if __name__ == "__main__":
    # 1. Clone the repo: git clone https://github.com/redcanaryco/atomic-red-team.git
    # 2. Update the path below to point to the 'atomics' folder
    path_to_atomics = "atomic-red-team/atomics"

    results = scrape_atomic_red_team(path_to_atomics)
    if results:
        save_to_atlas(results)

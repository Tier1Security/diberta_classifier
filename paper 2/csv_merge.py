import pandas as pd
import glob
import os

def merge_baselines(output_file="benign_baseline.csv", pattern="benign_baseline_*.csv"):
    """
    Merges multiple baseline CSV files from different machines into one master file.
    Expects files named like: benign_baseline_machine1.csv, benign_baseline_machine2.csv
    """
    all_files = glob.glob(pattern)

    if not all_files:
        print(f"[!] No files found matching pattern: {pattern}")
        return

    print(f"[*] Found {len(all_files)} baseline files. Merging...")

    combined_df = pd.concat([pd.read_csv(f) for f in all_files])

    # Remove duplicates to keep the training set efficient
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['command'])
    final_count = len(combined_df)

    combined_df.to_csv(output_file, index=False)

    print(f"[+] Merged {initial_count} events into {final_count} unique commands.")
    print(f"[+] Master baseline saved as: {output_file}")

if __name__ == "__main__":
    # Place all your collected CSVs in the same folder as this script
    merge_baselines()

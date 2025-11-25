from transformers import pipeline

# Load your merged model
classifier = pipeline(
    "text-classification", 
    model="models/merged_multiclass_roberta", 
    tokenizer="models/merged_multiclass_roberta"
)

# --- THE STRESS TEST ---
# These examples sit in the "Gray Area" between your classes
examples = [
    # 1. HARD NEGATIVE: Benign Reg save (Looks like T1003, but is safe key)
    "reg.exe save HKLM\\Software\\MyCorp\\Settings C:\\Temp\\settings.bkp",
    
    # 2. HARD NEGATIVE: Benign Firewall (Looks like T1562 structure, but standard port)
    "rule: Allow Web | app: C:\\Apache\\bin\\httpd.exe | dir: Inbound | act: Allow | port: 80",
    
    # 3. ADVERSARIAL: Malicious using "Safe" words
    "rule: Chrome Update | app: C:\\Users\\Public\\chrome_patch.exe | dir: Outbound | act: Allow | port: 443",
    
    # 4. AMBIGUOUS: "Whoami" (Should be benign, but often flagged)
    "cmd.exe /c whoami /all",
]

print("--- Stress Test Results ---")
for ex in examples:
    result = classifier(ex)[0]
    print(f"\nInput: {ex}")
    print(f"Pred:  {result['label']} ({result['score']:.4f})")
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pathlib
import torch.nn.functional as F
import re
from src.normalization import normalize_payload

# --- CONFIGURATION ---
# UPDATED: Pointing to the new 4-class merged model
TARGET_MODEL_PATH = "models/merged_4class_roberta"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)


def normalize_payload(obj):
    """Recursively normalize payload: lowercase, collapse whitespace.

    - Strings: trim, collapse whitespace to single space, lowercase
    - Maps: lower keys (strings) and normalize values recursively
    - Lists: normalize members recursively
    - Other values: returned unchanged
    """
    if isinstance(obj, str):
        return re.sub(r"\s+", " ", obj).strip().lower()
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            key = k.lower() if isinstance(k, str) else k
            new[key] = normalize_payload(v)
        return new
    if isinstance(obj, list):
        return [normalize_payload(i) for i in obj]
    return obj

# --- 1. LOAD MODEL & TOKENIZER ---
print(f"--- Initializing Security AI API (4-Class) ---")
print(f"Target Device: {DEVICE}")

try:
    model_path = pathlib.Path(TARGET_MODEL_PATH)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found at: {TARGET_MODEL_PATH}")

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    print(f"Loading model weights from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode

    # Extract label mappings from the model config
    # This will now automatically include "T1134" (Label 3)
    id2label = model.config.id2label
    label2id = model.config.label2id
    print(f"Model Labels: {id2label}")
    print("--- Initialization Complete ---\n")

except Exception as e:
    print(f"\n[FATAL ERROR] Could not load model: {e}")
    print(f"Ensure '{TARGET_MODEL_PATH}' exists and contains config.json/pytorch_model.bin.")
    exit(1)

# --- 2. PREDICTION ENDPOINT ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON: { "text": "reg save HKLM\\SAM ..." }

    NOTE: All incoming string payloads (JSON values, form fields, or raw body text)
    are automatically normalized:
    - lowercased
    - whitespace collapsed to a single space, leading/trailing trimmed
    This ensures consistent preprocessing regardless of incoming casing/spacing.
    """
    
    # A. Extract & Normalize Input
    # Use top-level normalize function

    text_input = None
    if request.is_json:
        # get a parsed JSON payload safely
        raw_json = request.get_json(silent=True)
        if raw_json is None:
            # fallback: try request.json (may raise) or raw data
            try:
                raw_json = request.json
            except Exception:
                raw_json = None
        if raw_json is not None:
            normalized_json = normalize_payload(raw_json)
            if isinstance(normalized_json, dict):
                # normalized keys are lowercased so 'text' will match 'Text', 'TEXT', etc.
                text_input = normalized_json.get("text")
            elif isinstance(normalized_json, str):
                text_input = normalized_json
    elif request.form:
        raw_form_value = request.form.get("text")
        if raw_form_value:
            text_input = normalize_payload(raw_form_value)
    else:
        data = request.get_data(as_text=True)
        if data:
            text_input = normalize_payload(data)

    if not text_input:
        return jsonify({"error": "No text provided"}), 400

    # B. Preprocess (text already lowercased and whitespace-normalized from the input stage)
    processed_text = text_input
    
    # C. Inference
    try:
        inputs = tokenizer(
            processed_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128,
            padding="max_length"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Calculate Probabilities
        probs = F.softmax(logits, dim=1)[0]
        
        # Get the top prediction
        pred_id = torch.argmax(probs).item()
        pred_label = id2label[pred_id]
        pred_conf = probs[pred_id].item()

        # Build detailed score dictionary
        scores = {
            label: round(probs[i].item(), 4) 
            for i, label in id2label.items()
        }

        return jsonify({
            "label": pred_label,
            "confidence score": round(pred_conf, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 3. HEALTH CHECK ---
@app.route("/", methods=["GET"])
def index():
    return f"""
    <html>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1>🛡️ Security AI Classifier Online</h1>
            <p>Model: <b>RoBERTa 4-Class (Security)</b></p>
            <p>Labels: {list(id2label.values())}</p>
            <p>Status: <span style="color: green;">Ready</span></p>
        </body>
    </html>
    """



def kill_processes_on_port(port: int, timeout: float = 2.0):
    """Attempt to gracefully terminate processes listening on `port`, then SIGKILL if needed.

    Note: This function is Linux-focused and may require privileges to kill some processes.
    """
    # Function removed: no-op placeholder to preserve API if referenced elsewhere.
    return


if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=80)
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pathlib
import torch.nn.functional as F

# --- CONFIGURATION ---
# UPDATED: Pointing to the new 4-class merged model
TARGET_MODEL_PATH = "models/merged_4class_roberta"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

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
    """
    
    # A. Extract Input
    text_input = None
    if request.is_json:
        text_input = request.json.get("text")
    elif request.form:
        text_input = request.form.get("text")
    else:
        data = request.get_data(as_text=True)
        if data: text_input = data

    if not text_input:
        return jsonify({"error": "No text provided"}), 400

    # B. Preprocess
    processed_text = text_input.lower()
    
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
    
    app.run(host='127.0.0.1', port=80)
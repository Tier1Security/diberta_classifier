from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pathlib
import torch.nn.functional as F

# --- CONFIGURATION ---
# We prioritize the path used in the merge_3class.py script
TARGET_MODEL_PATH = "models/merged_multiclass_roberta"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)

# --- 1. LOAD MODEL & TOKENIZER ---
print(f"--- Initializing Security AI API ---")
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
    model.eval() # Set to evaluation mode (disable dropout, etc)

    # Extract label mappings from the model config
    id2label = model.config.id2label
    label2id = model.config.label2id
    print(f"Model Labels: {id2label}")
    print("--- Initialization Complete ---\n")

except Exception as e:
    print(f"\n[FATAL ERROR] Could not load model: {e}")
    print("Did you run 'merge_3class.py'? Ensure 'models/merged_multiclass_roberta' exists.")
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
        # Fallback for raw string body
        data = request.get_data(as_text=True)
        if data: text_input = data

    if not text_input:
        return jsonify({"error": "No text provided"}), 400

    # B. Preprocess (CRITICAL: Match Training Logic)
    # We must lowercase because the training generator used normalize_case()
    processed_text = text_input.lower()
    
    # C. Inference
    try:
        # Tokenize
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
        
        # Calculate Probabilities (Softmax)
        probs = F.softmax(logits, dim=1)[0]
        
        # Get the top prediction
        pred_id = torch.argmax(probs).item()
        pred_label = id2label[pred_id]
        pred_conf = probs[pred_id].item()

        # Build detailed score dictionary for all classes
        scores = {
            label: round(probs[i].item(), 4) 
            for i, label in id2label.items()
        }

        response = {
            "input": text_input[:200], # Echo back (truncated)
            "verdict": pred_label,
            "confidence": round(pred_conf, 4),
            "scores": scores # Full breakdown for SIEM/Logging
        }
        
        # Optional: Add a "high_risk" flag for the client
        if pred_label != "Benign" and pred_conf > 0.90:
            response["alert"] = True
        else:
            response["alert"] = False

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 3. HEALTH CHECK ---
@app.route("/", methods=["GET"])
def index():
    return f"""
    <html>
        <body style="font-family: sans-serif; text-align: center; padding-top: 50px;">
            <h1>🛡️ Security AI Classifier Online</h1>
            <p>Model: <b>RoBERTa Multi-Class (Security)</b></p>
            <p>Labels: {list(id2label.values())}</p>
            <p>Status: <span style="color: green;">Ready</span></p>
        </body>
    </html>
    """

if __name__ == "__main__":
    # Bind to localhost only for safety (prevent external access).
    app.run(host='127.0.0.1', port=5050)
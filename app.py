from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pathlib

# --- CONFIGURATION ---
# IMPORTANT: The model directory must contain the merged weights and configuration 
# (e.g., pytorch_model.bin, config.json, vocab.json, tokenizer_config.json).
# This script will attempt to find a folder starting with 'merged' under 'models/'.
models_dir = pathlib.Path("models")
MERGED_MODEL_PATH = None

# Attempt to auto-detect the merged model path
if models_dir.exists():
    # Prefer directories that contain model files (config/weights/tokenizer)
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    for p in models_dir.iterdir():
        if not p.is_dir():
            continue
        if any((p / f).exists() for f in required_files):
            MERGED_MODEL_PATH = str(p)
            break
    # Fallback: find directories with 'merged' or 'final' in the name
    if MERGED_MODEL_PATH is None:
        for p in models_dir.iterdir():
            if p.is_dir() and (p.name.startswith("merged") or "merged" in p.name or "final" in p.name):
                MERGED_MODEL_PATH = str(p)
                break

if MERGED_MODEL_PATH is None:
    # Set a fallback path if auto-detection fails
    MERGED_MODEL_PATH = "models/merged_roberta_model"
    print(f"Warning: No 'merged*' model directory found under 'models/'. Falling back to {MERGED_MODEL_PATH}")

print(f"Using model directory: {MERGED_MODEL_PATH}")

# Set device for inference
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. LOAD THE TRAINED MODEL AND TOKENIZER ---
print(f"Loading merged model and tokenizer from: {MERGED_MODEL_PATH}...")
try:
    # Use bfloat16 for efficient GPU inference if available, otherwise standard float32
    dtype = (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    
    # Load the full, merged model using standard AutoModel class
    # Ensure we only load local files (avoid trying to pull from the hub)
    model_path = pathlib.Path(MERGED_MODEL_PATH)
    # Check for common model files before attempting to load
    required_files = [
        "config.json",
        "pytorch_model.bin",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    ]
    has_any = any((model_path / f).exists() for f in required_files)
    if not model_path.exists() or not model_path.is_dir() or not has_any:
        existing = []
        if model_path.exists() and model_path.is_dir():
            existing = [p.name for p in model_path.iterdir()]
        print(f"FATAL ERROR: Local merged model directory '{MERGED_MODEL_PATH}' is missing or has no expected model files.")
        print(f"Expected one of: {required_files}")
        print(f"Files found: {existing}")
        raise FileNotFoundError(f"Merged model not found or incomplete at {MERGED_MODEL_PATH}")

    model = AutoModelForSequenceClassification.from_pretrained(str(model_path), torch_dtype=dtype, local_files_only=True)
    model.eval() # Set model to evaluation mode
    model = model.to(DEVICE)
    
    # Load the corresponding tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    
    print(f"Model loaded successfully on device: {DEVICE}")
    
    # The id2label mapping is saved in the model's config
    id2label = model.config.id2label

except Exception as e:
    print(f"FATAL ERROR: Failed to load merged model from {MERGED_MODEL_PATH}.")
    print("Please ensure the directory exists and contains all necessary Hugging Face files (e.g., config.json, pytorch_model.bin).")
    print(f"Details: {e}")
    # Raise the error to stop the application from starting
    raise

# --- 2. CREATE FLASK APP ---
app = Flask(__name__)

# --- 3. DEFINE THE PREDICTION ENDPOINT ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a POST request with a JSON body containing 'text'
    and returns the model's prediction for that text.
    """
    
    # Robustly extract the text input from the request
    text_input = None

    # 1. Attempt JSON body {"text": "..."}
    if request.is_json:
        try:
            data = request.get_json(silent=True)
            if isinstance(data, dict):
                text_input = data.get("text")
        except Exception:
            text_input = None

    # 2. Next try form data
    if not text_input and request.form:
        text_input = request.form.get("text")

    # 3. Finally, accept raw body as a fallback
    if not text_input:
        raw = request.get_data(as_text=True)
        if raw:
            # Simple fallback for text field if JSON parsing failed
            if raw.strip().startswith("{") and "\"text\"" in raw:
                 try:
                    # Crude extraction of the string value for 'text'
                    start = raw.find('"text"')
                    colon = raw.find(':', start)
                    first_quote = raw.find('"', colon + 1)
                    second_quote = raw.find('"', first_quote + 1)
                    if first_quote != -1 and second_quote != -1:
                        text_input = raw[first_quote + 1:second_quote]
                    else:
                        text_input = raw
                 except Exception:
                    text_input = raw
            else:
                text_input = raw

    if not text_input:
        return jsonify({"error": "Missing 'text' field in request body. Use JSON: {'text': 'command'} or send raw text."}), 400
    # CRITICAL CHANGE: Normalize the input to lowercase before feeding to the model
    processed_input = text_input.lower()
    print(f"Normalized input (lowercase): '{processed_input}'")

    # Tokenize the processed (lowercased) text
    inputs = tokenizer(processed_input, return_tensors="pt", max_length=128, padding="max_length", truncation=True)

    # --- 4. PREPARE INPUT FOR THE MODEL ---

    # Move tensors to the same device as the model
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    # --- 5. GET PREDICTION ---
    with torch.no_grad():
        logits = model(**inputs).logits

    # --- 6. PROCESS THE OUTPUT ---
    predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = id2label[predicted_class_id]
    
    # Calculate probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
    
    # Get the confidence for the predicted label
    predicted_probability = probabilities[predicted_class_id].item()
    
    # Optionally, get the confidence for the specific malicious label (T1003.002)
    malicious_label_id = model.config.label2id.get("T1003.002")
    t1003_confidence = probabilities[malicious_label_id].item() if malicious_label_id is not None else "N/A"

    print(f"Prediction: '{predicted_label}' | Confidence: {predicted_probability:.4f} | T1003 Score: {t1003_confidence}")

    # --- 7. RETURN THE RESULT ---
    return jsonify({
        "input_text": text_input,
        "predicted_label": predicted_label,
        "confidence_score": f"{predicted_probability:.4f}",
        "T1003.002_score": f"{t1003_confidence:.4f}"
    })

# --- 8. DEFINE A SIMPLE ROOT ENDPOINT ---
@app.route("/", methods=["GET"])
def index():
    return """
    <html>
    <head>
        <title>DeBERTa Threat Classifier API</title>
        <style>
            body { font-family: sans-serif; padding: 20px; }
            code { background-color: #eee; padding: 2px 4px; border-radius: 3px; }
            h2 { color: #333; }
        </style>
    </head>
    <body>
        <h2>DeBERTa Threat Classifier API</h2>
        <p>This API provides classification for Windows command-line arguments to detect the T1003.002 credential dumping technique (Registry Save).</p>
        
        <h3>Usage</h3>
        <p>Send a <code>POST</code> request to the <code>/predict</code> endpoint with a JSON body:</p>
        <pre><code>
{
    "text": "reg save HKLM\\SAM C:\\Windows\\Temp\\sam.hiv"
}
        </code></pre>
        <p>The API will return the predicted label (e.g., 'T1003.002' or 'Benign') and confidence scores.</p>
    </body>
    </html>
    """

if __name__ == "__main__":
    # Run the app on all available network interfaces
    app.run(host='0.0.0.0', port=8000, debug=True)
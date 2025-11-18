from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import pathlib

# --- CONFIGURATION ---
# Use the merged model for straightforward deployment
# Auto-detect available merged model under `models/` (prefers any folder starting with 'merged')
models_dir = pathlib.Path("models")
selected_model = None
if models_dir.exists():
    for p in models_dir.iterdir():
        if p.is_dir() and p.name.startswith("merged"):
            selected_model = p
            break

if selected_model is None:
    # Fallback path used previously in the repository
    MERGED_MODEL_PATH = "models/merged_deberta_model"
    print(f"No 'merged*' model directory found under 'models/'. Falling back to {MERGED_MODEL_PATH}")
else:
    MERGED_MODEL_PATH = str(selected_model)
    print(f"Using merged model directory: {MERGED_MODEL_PATH}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. LOAD THE TRAINED MODEL AND TOKENIZER ---
print(f"Loading merged model and tokenizer from: {MERGED_MODEL_PATH}...")
try:
    dtype = (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForSequenceClassification.from_pretrained(MERGED_MODEL_PATH, torch_dtype=dtype)
    model = model.to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)
    print(f"Model loaded successfully on device: {DEVICE}")
except Exception as e:
    print(f"Failed to load merged model from {MERGED_MODEL_PATH}: {e}")
    raise

# The id2label mapping is saved in the model's config, so we can get it from there
id2label = model.config.id2label

# --- 2. CREATE FLASK APP ---
app = Flask(__name__)

# --- 3. DEFINE THE PREDICTION ENDPOINT ---
@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a POST request with a JSON body containing 'text'
    and returns the model's prediction for that text.
    """
    # Try several ways to extract text safely:
    # 1) Proper JSON body {"text": "..."}
    # 2) Form data (multipart/form-data or application/x-www-form-urlencoded)
    # 3) Raw text in the request body (useful when JSON escaping is problematic)

    text_input = None

    # Attempt JSON first (if content-type is application/json)
    if request.is_json:
        try:
            data = request.get_json(silent=True)
            if isinstance(data, dict):
                text_input = data.get("text")
        except Exception:
            text_input = None

    # Next try form data
    if not text_input and request.form:
        text_input = request.form.get("text")

    # Finally, accept raw body as a fallback (treat entire body as the text)
    if not text_input:
        raw = request.get_data(as_text=True)
        if raw:
            # If the raw body looks like a JSON object (but JSON decoding failed due to escapes),
            # try a quick, lenient extraction of the value for the 'text' key.
            raw_stripped = raw.strip()
            if raw_stripped.startswith("{") and "\"text\"" in raw_stripped:
                # crude parsing: find first occurrence of "text" and extract following string
                try:
                    start = raw_stripped.index('"text"')
                    # find the first quote after the colon
                    colon = raw_stripped.index(':', start)
                    first_quote = raw_stripped.index('"', colon)
                    second_quote = raw_stripped.index('"', first_quote+1)
                    # extract without interpreting escape sequences
                    text_input = raw_stripped[first_quote+1:second_quote]
                except Exception:
                    # fallback: use entire raw body
                    text_input = raw
            else:
                text_input = raw

    if not text_input:
        return jsonify({"error": "Missing 'text' field in request body"}), 400

    print(f"\nReceived input for prediction: '{text_input}'")

    # --- 4. PREPARE INPUT FOR THE MODEL ---
    # Tokenize the input text
    inputs = tokenizer(text_input, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    
    # Move tensors to the same device as the model
    inputs = {key: val.to(DEVICE) for key, val in inputs.items()}

    # --- 5. GET PREDICTION ---
    # Perform inference without calculating gradients
    with torch.no_grad():
        logits = model(**inputs).logits

    # --- 6. PROCESS THE OUTPUT ---
    # Get the most likely prediction by finding the index with the highest logit score
    predicted_class_id = torch.argmax(logits, dim=1).item()
    
    # Get the corresponding label name
    predicted_label = id2label[predicted_class_id]
    
    # Get the probability score of the predicted class (using softmax)
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_probability = probabilities[0][predicted_class_id].item()

    print(f"Prediction: '{predicted_label}' with probability: {predicted_probability:.4f}")

    # --- 7. RETURN THE RESULT ---
    return jsonify({
        "input_text": text_input,
        "predicted_label": predicted_label,
        "confidence_score": predicted_probability
    })

# --- 8. DEFINE A SIMPLE ROOT ENDPOINT ---
@app.route("/")
def index():
    return "<h2>DeBERTa Threat Classifier API</h2><p>Send a POST request to /predict with a JSON body like {'text': 'your command here'}</p>"

if __name__ == "__main__":
    # Run the app on all available network interfaces
    app.run(host='0.0.0.0', port=8000, debug=True)

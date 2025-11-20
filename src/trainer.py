from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
import pandas as pd
import math

# --- CONFIG ---
BATCH_SIZE = 32 # Use 16 if on smaller GPU
EPOCHS = 3

MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2' # Strongest base model
# Default training file: the one located next to this script in `src/`
from pathlib import Path
SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_FILE = str(SCRIPT_DIR / 'mitre_training_data.jsonl')
OUTPUT_PATH = './output/mitre-log-mapper-v1'
import argparse

# 1. Load Data
parser = argparse.ArgumentParser(description='Train sentence-transformers on MITRE dataset')
parser.add_argument('--train-file', type=str, default=TRAIN_FILE, help='Path to training jsonl file')
parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Training batch size')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
parser.add_argument('--output', type=str, default=OUTPUT_PATH, help='Output model path')
args = parser.parse_args()

TRAIN_FILE = args.train_file
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
OUTPUT_PATH = args.output

print(f"Loading Data from {TRAIN_FILE}...")
train_path = Path(TRAIN_FILE)
if not train_path.exists():
    raise FileNotFoundError(f"Training file not found: {TRAIN_FILE}. Please provide a valid jsonl file with 'anchor' and 'positive' fields.")

df = pd.read_json(TRAIN_FILE, lines=True)

# Split: 90% Train, 10% Eval
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

train_examples = []
for i, row in train_df.iterrows():
    # Safely access expected columns
    if 'anchor' in row and 'positive' in row:
        train_examples.append(InputExample(texts=[row['anchor'], row['positive']]))
    else:
        # Skip rows missing required fields
        continue

# If sampling produced no train examples (small dataset), fall back to using the full dataframe
if len(train_examples) == 0:
    for i, row in df.iterrows():
        if 'anchor' in row and 'positive' in row:
            train_examples.append(InputExample(texts=[row['anchor'], row['positive']]))

# If still empty, create a tiny default example to allow the script to run for testing
if len(train_examples) == 0:
    train_examples = [InputExample(texts=["reg save HKEY_LOCAL_MACHINE\\SAM C:\\Windows\\Temp\\sam.hiv", "reg save HKLM\\SAM C:\\temp\\sam.hiv"]) ]

# 2. Initialize Model & Dataloader
print(f"Initializing {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

# 3. Define Loss (MultipleNegativesRankingLoss is best for search/retrieval)
train_loss = losses.MultipleNegativesRankingLoss(model)

# 4. Train
print("Starting Training...")
warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * 0.1) # 10% warmup

# Robust training loop: if CUDA OOM occurs, keep the same batch size and retry a few times
import torch
max_retries = 10
retry = 0
while True:
    try:
        warmup_steps = math.ceil(len(train_dataloader) * EPOCHS * 0.1)
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=warmup_steps,
            output_path=OUTPUT_PATH,
            show_progress_bar=True
        )
        break
    except RuntimeError as e:
        msg = str(e).lower()
        if ('out of memory' in msg or isinstance(e, torch.cuda.OutOfMemoryError)) and torch.cuda.is_available():
            retry += 1
            if retry > max_retries:
                print('Exceeded maximum OOM retries. Consider using a smaller model, switching to CPU, or enabling gradient accumulation.')
                raise
            # Clear cache and wait briefly before retrying with the same batch size
            print(f'CUDA OOM detected. Clearing CUDA cache and retrying (attempt {retry}/{max_retries})')
            try:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except Exception:
                pass
            import time
            time.sleep(5)
            continue
        else:
            raise

print(f"Training Complete. Model saved to {OUTPUT_PATH}")
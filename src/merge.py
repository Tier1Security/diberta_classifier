import pickle
import pandas as pd
from sentence_transformers import SentenceTransformer
import json

# --- CONFIG ---
MODEL_PATH = './output/mitre-log-mapper-v1'
DATA_FILE = 'mitre_training_data.jsonl'
DEPLOY_DIR = './deploy_package'

import os
if not os.path.exists(DEPLOY_DIR):
    os.makedirs(DEPLOY_DIR)

print("1. Loading Resources...")
model = SentenceTransformer(MODEL_PATH)
df = pd.read_json(DATA_FILE, lines=True)

# Get unique Techniques (Code + Name + Description)
# We group by 'positive' text to ensure 1 vector per technique definition
unique_techniques = df.groupby('positive').first().reset_index()

print(f"2. Pre-computing Embeddings for {len(unique_techniques)} unique techniques...")
embeddings = model.encode(unique_techniques['positive'].tolist(), show_progress_bar=True)

# 3. Exporting Artifacts
print("3. Saving Deployable Artifacts...")

# Save the vectors (The "Index")
with open(f'{DEPLOY_DIR}/mitre_index.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Save the Metadata (The "Lookup Table")
unique_techniques[['t_code', 'positive']].to_pickle(f'{DEPLOY_DIR}/mitre_metadata.pkl')

# Save the Model itself (copying it over)
model.save(f'{DEPLOY_DIR}/model')

print(f"\nDONE! Your deployment package is ready at '{DEPLOY_DIR}'")
print("You can now load this package in your production pipeline without re-training.")
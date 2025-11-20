from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch

# --- CONFIG ---
MODEL_PATH = './output/mitre-log-mapper-v1'
DATA_FILE = 'mitre_training_data.jsonl'

print("Loading Fine-Tuned Model...")
model = SentenceTransformer(MODEL_PATH)

# Load Data (Re-splitting to get the same test set)
df = pd.read_json(DATA_FILE, lines=True)
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

# 1. Embed the entire MITRE Database (The "Knowledge Base")
# We use the unique Positive descriptions as our search index
unique_techniques = df['positive'].unique()
print(f"Encoding Knowledge Base ({len(unique_techniques)} techniques)...")
kb_embeddings = model.encode(unique_techniques, convert_to_tensor=True)

# 2. Run Evaluation
print("Running Evaluation on Test Set...")
correct_at_1 = 0
correct_at_5 = 0
correct_at_10 = 0
total = 0

for i, row in test_df.iterrows():
    # Embed the query (Log)
    query_emb = model.encode(row['anchor'], convert_to_tensor=True)
    
    # Search
    hits = util.semantic_search(query_emb, kb_embeddings, top_k=10)[0]
    
    # Check if correct answer is in results
    found_indices = [hit['corpus_id'] for hit in hits]
    found_texts = [unique_techniques[idx] for idx in found_indices]
    
    if row['positive'] == found_texts[0]:
        correct_at_1 += 1
    if row['positive'] in found_texts[:5]:
        correct_at_5 += 1
    if row['positive'] in found_texts[:10]:
        correct_at_10 += 1
    total += 1

print("-" * 30)
print(f"Total Test Queries: {total}")
print(f"Accuracy @ 1:  {correct_at_1 / total:.2%}")
print(f"Recall   @ 5:  {correct_at_5 / total:.2%}")
print(f"Recall   @ 10: {correct_at_10 / total:.2%}")
print("-" * 30)
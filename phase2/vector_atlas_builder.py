import pandas as pd
import torch
import faiss
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

def build_mitre_vector_index(csv_input="mitre_atlas_raw.csv", index_output="mitre_atlas.index"):
    """
    Converts the raw MITRE CSV into a searchable FAISS vector index.
    This index serves as the "Dictionary of Evil" for your attribution logic.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    encoder = RobertaModel.from_pretrained("roberta-base").to(device).eval()
    
    # Check if the raw data exists
    try:
        df = pd.read_csv(csv_input)
    except FileNotFoundError:
        print(f"[!] {csv_input} not found. Please run the MITRE Atlas Scraper first.")
        return

    print(f"[*] Vectorizing {len(df)} MITRE templates into a 768-D space...")
    
    vectors = []
    
    with torch.no_grad():
        for cmd in tqdm(df['command']):
            # Ensure the command is a string and handle potential NaNs
            text = str(cmd).lower()
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = encoder(**inputs)
            
            # Use the CLS token embedding (first token) as the sentence-level representation
            vec = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            vectors.append(vec)
            
    # Convert list of vectors to a single NumPy array
    vectors = np.vstack(vectors).astype('float32')
    
    # Create the FAISS Index using L2 (Euclidean) Distance
    # d = 768 is the standard hidden dimension for RoBERTa-base
    d = 768 
    index = faiss.IndexFlatL2(d)
    index.add(vectors)
    
    # Save the binary index file for use in inference
    faiss.write_index(index, index_output)
    
    # Save a metadata CSV to map index positions back to MITRE IDs and names
    df[['mitre_id', 'technique_name', 'command']].to_csv("mitre_atlas_metadata.csv", index=False)
    
    print(f"[+] Vector Atlas built successfully!")
    print(f"[+] Index file: {index_output}")
    print(f"[+] Metadata file: mitre_atlas_metadata.csv")

if __name__ == "__main__":
    # Ensure mitre_atlas_raw.csv is present in the directory
    # build_mitre_vector_index()
    pass
import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, EsmModel

# The storage location of the ESM model
MODEL_NAME = "../ESM/"

# Input files and output directories
INPUT_FASTA = "../data/targetp/targetp.fasta"
OUTPUT_DIR = "../data/targetp/ESM_data/"

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load the model
print(f"Loading model: {MODEL_NAME} ...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
except Exception as e:
    print("Failed to load the model. Please check the network connection or the file path.")
    raise e

print("Reading FASTA file...")
names = []
sequences = []

with open(INPUT_FASTA, 'r') as f:
    lines = f.read().splitlines()
    id_lines = lines[0::2]
    seq_lines = lines[1::2]

    for i in range(len(seq_lines)):
        name = id_lines[i].strip().lstrip(">")
        seq = seq_lines[i].strip()

        if len(seq) > 200:
            seq = seq[:200]

        names.append(name)
        sequences.append(seq)

print(f"Total sequences: {len(sequences)}")

print("Start extracting features...")

with torch.no_grad():
    for i, seq in enumerate(sequences):
        name = names[i]

        inputs = tokenizer(seq, return_tensors="pt", padding=False, truncation=True, max_length=202)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model(**inputs)

        raw_emb = outputs.last_hidden_state.squeeze(0)
        seq_emb = raw_emb[1:-1, :]

        current_len = seq_emb.shape[0]
        dim = seq_emb.shape[1]
        seq_emb_np = seq_emb.cpu().numpy()
        final_emb = np.zeros((200, dim), dtype=np.float32)

        if current_len >= 200:
            final_emb = seq_emb_np[:200, :]
        else:
            final_emb[:current_len, :] = seq_emb_np

        df = pd.DataFrame(final_emb)
        save_path = os.path.join(OUTPUT_DIR, f"{name}.feather")
        df.to_feather(save_path)

        if (i + 1) % 100 == 0:
            print(f"\rProgress: {i + 1}/{len(sequences)}", end="", flush=True)

print(f"\nDone! Features saved to {OUTPUT_DIR}")
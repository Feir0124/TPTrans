import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

# The main function of ISM.py is to utilize a pre-trained ISM (Implicit Structural Model), read protein sequence files in FASTA format,
# convert them into high-dimensional vector representations (Embeddings), and save these features to the hard disk in .feather format for subsequent TPTrans model training.

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

names = []
number_sequences = []

with open("../data/targetp/targetp.fasta", 'r') as f:
    lines = f.read().splitlines()
    flag = lines[0::2]
    sequences = lines[1::2]

    for i in range(len(sequences)):
        name = flag[i][1:].strip()
        names.append(name.lstrip(">"))

        if len(sequences[i]) < 200:
            number_sequence = sequences[i]
            for _ in range(200-len(sequences[i])):
                number_sequence+='<pad>'
            number_sequences.append(number_sequence)
        else:
            number_sequences.append(sequences[i][:200])

config_path = "../ISM/"
model = AutoModel.from_pretrained(config_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(config_path)

num = 0
print('start')

for i in range(len(number_sequences)):
    input_text = number_sequences[i]

    batch_labels = tokenizer(input_text, return_tensors="pt").to("cuda")

    out = model(batch_labels['input_ids'].to("cuda"),attention_mask=batch_labels['attention_mask'].to("cuda"))

    raw_emb = out.last_hidden_state.detach().cpu().numpy().squeeze(0)

    if raw_emb.shape[0] == 202:
        clean_emb = raw_emb[1:-1, :]
    else:
        clean_emb = raw_emb

    df = pd.DataFrame(clean_emb)

    print(df.shape)

    df.to_feather('../data/targetp/ISM_data/'+names[i]+'.feather')

    num += 1
    print(f'\rProgress: {num}', end='', flush=True)
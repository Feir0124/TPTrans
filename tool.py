import random
import numpy as np
import pandas as pd
import torch
import os

from pygments.lexer import combined
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


species_to_int = {'Other': 0, 'SP': 1, 'MT': 2, 'CH': 3, 'TH': 4}
amino_acid_to_int = {'X': 0, 'M': 1, 'I': 2, 'L': 3, 'S': 4, 'H': 5, 'R': 6, 'P': 7, 'A': 8,
                             'W': 9, 'F': 10, 'D': 11, 'C': 12, 'T': 13, 'N': 14, 'V': 15, 'G': 16,
                             'Q': 17, 'K': 18, 'Y': 19, 'E': 20, 'U': 21, 'Z':22, 'B': 23}

path2 = 'targetp.tab'
fasta_path = 'targetp.fasta'

BLOSUM62_AA = "ARNDCQEGHILKMFPSTWYVBZX*"
B62_DICT = {aa: i for i, aa in enumerate(BLOSUM62_AA)}
BLOSUM62_MATRIX = np.array([
    [ 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4],
    [-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4],
    [-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4],
    [-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4],
    [ 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4],
    [-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4],
    [-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
    [ 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4],
    [-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4],
    [-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4],
    [-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4],
    [-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4],
    [-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4],
    [-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4],
    [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4],
    [ 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4],
    [ 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4],
    [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4],
    [-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4],
    [ 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4],
    [-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4],
    [-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4],
    [ 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4],
    [-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1]
], dtype=np.float32)
BLOSUM62_MATRIX = (BLOSUM62_MATRIX - np.min(BLOSUM62_MATRIX)) / (np.max(BLOSUM62_MATRIX) - np.min(BLOSUM62_MATRIX))


# Convert the sequence to a one-hot matrix
def seq_to_one_hot(seq_str, max_len=200):
    if len(seq_str) > max_len:
        seq_str = seq_str[:max_len]

    one_hot = np.zeros((max_len, 24), dtype=np.float32)

    for i, aa in enumerate(seq_str):
        idx = amino_acid_to_int.get(aa.upper(), 0)
        one_hot[i, idx] = 1.0

    return one_hot


# Convert the sequence to a BLOSUM62 matrix
def seq_to_blosum62(seq_str, max_len=200):
    if len(seq_str) > max_len:
        seq_str = seq_str[:max_len]

    pad_vector = BLOSUM62_MATRIX[-1]
    feature_matrix = np.tile(pad_vector, (max_len, 1))

    for i, aa in enumerate(seq_str):
        idx = B62_DICT.get(aa.upper(), 22)
        feature_matrix[i] = BLOSUM62_MATRIX[idx]

    return feature_matrix


# --- Set the random seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --- Dataset category ---
class TPDataset(Dataset):
    def __init__(self, names, sequences, tags, species, is_train=False, feature_type='TPTrans'):
        self.names = names
        self.sequences = sequences
        self.tags = tags
        self.species = species
        self.is_train = is_train
        self.feature_type = feature_type

        self.ism_dir = 'ISM_data'
        self.esm_dir = 'ESM_data'

    def __len__(self):
        return len(self.names)

    # --- Random Mask ---
    def apply_masking(self, feature, tag):
        tag_np = np.array(tag)

        ones_indices = np.where(tag_np == 1)[0]
        if len(ones_indices) == 0:
            cs_idx = np.random.randint(20, 100)
        else:
            cs_idx = ones_indices[-1]

        safe_margin = 2
        safe_start = max(0, cs_idx - safe_margin)
        safe_end = min(200, cs_idx + safe_margin + 1)

        mask_noise_zone = torch.zeros(200, 1)
        prob = np.random.rand()

        if prob < 0.3:
            if safe_start > 2:
                mask_len = np.random.randint(1, safe_start)
                mask_noise_zone[0: mask_len, :] = 1

        elif prob < 0.6:
            if safe_end < 198:
                mask_start = np.random.randint(safe_end, 199)
                mask_noise_zone[mask_start: 200, :] = 1

        elif prob < 0.8:
            block_len = np.random.randint(5, 20)
            start = np.random.randint(0, 200 - block_len)

            if (start + block_len) <= safe_start or start >= safe_end:
                mask_noise_zone[start: start + block_len, :] = 1

        noise = torch.randn_like(feature) * 0.2

        augmented_feature = feature + (noise * mask_noise_zone)

        return augmented_feature

    def __getitem__(self, idx):
        name = self.names[idx]
        seq_str = self.sequences[idx]

        if self.feature_type == 'One-hot':
            # One-hot
            feat_matrix = seq_to_one_hot(seq_str, max_len=200)

        elif self.feature_type == 'BLOSUM62':
            # BLOSUM62
            feat_matrix = seq_to_blosum62(seq_str, max_len=200)

        elif self.feature_type == 'ISM':
            # Only ISM
            ism_path = os.path.join(self.ism_dir, f'{name}.feather')
            feat_matrix = pd.read_feather(ism_path).to_numpy()
            if feat_matrix.shape[0] == 202: feat_matrix = feat_matrix[1:-1, :]

        elif self.feature_type == 'ESM2':
            # Only ESM2
            esm_path = os.path.join(self.esm_dir, f'{name}.feather')
            feat_matrix = pd.read_feather(esm_path).to_numpy()
            if feat_matrix.shape[0] == 202: feat_matrix = feat_matrix[1:-1, :]

        elif self.feature_type == 'TPTrans':
            ism_path = os.path.join(self.ism_dir, f'{name}.feather')
            ism_feat = pd.read_feather(ism_path).to_numpy()
            if ism_feat.shape[0] == 202: ism_feat = ism_feat[1:-1, :]

            esm_path = os.path.join(self.esm_dir, f'{name}.feather')
            esm_feat = pd.read_feather(esm_path).to_numpy()
            if esm_feat.shape[0] == 202: esm_feat = esm_feat[1:-1, :]

            feat_matrix = np.concatenate([ism_feat, esm_feat], axis=1)

        else:
            raise ValueError(f"Unknown feature type: {self.feature_type}")

        feature_tensor = torch.tensor(feat_matrix, dtype=torch.float32)

        tag_item = self.tags[idx]
        tag_list = tag_item.tolist() if isinstance(tag_item, np.ndarray) else list(tag_item)
        tag_tensor = torch.tensor(tag_list, dtype=torch.long)

        species_label = self.species[idx]
        species_tensor = torch.as_tensor(species_label, dtype=torch.long)

        if self.is_train:
            feature_tensor = self.apply_masking(feature_tensor, tag_list)

        return feature_tensor, tag_tensor, species_tensor


# --- Read the label file ---
def load_data():
    name_sequences = []
    tags_dictionary = {}
    species_dictionary = {}
    seq_dictionary = {}

    print(f"Loading annotations from {path2}...")
    with open(path2, 'r') as f:
        for line in f:
            columns = line.strip().split('\t')

            tag = []
            cs_len = int(columns[2])
            for _ in range(cs_len):
                tag.append(1)
            for _ in range(200 - cs_len):
                tag.append(0)

            tags_dictionary[columns[0]] = tag
            species_dictionary[columns[0]] = species_to_int.get(columns[1], 0)  # 默认为0(Other)
            name_sequences.append(columns[0])

    print(f"Loading actual sequences from {fasta_path}...")
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    seq_dictionary[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
        if current_id is not None:
            seq_dictionary[current_id] = "".join(current_seq)

    final_names, final_seqs, final_tags, final_species = [], [], [], []
    missing_count = 0
    for name in name_sequences:
        if name in seq_dictionary:
            final_names.append(name)
            final_seqs.append(seq_dictionary[name])
            final_tags.append(tags_dictionary[name])
            final_species.append(species_dictionary[name])
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} IDs in the tab file failed to find their corresponding sequences in the fasta file!")

    return final_names, final_seqs, final_tags, final_species


# --- Create and partition the dataset ---
def creat_data(feature_type='TPTrans'):
    name, seqs, tag, species = load_data()

    fold_size = len(name) // 5

    def get_fold(i):
        start = i * fold_size
        end = (i + 1) * fold_size if i < 4 else len(name)
        return name[start:end], seqs[start:end], tag[start:end], species[start:end]

    f1 = get_fold(0);
    f2 = get_fold(1);
    f3 = get_fold(2);
    f4 = get_fold(3);
    f5 = get_fold(4)

    data1 = TPDataset(f1[0], f1[1], f1[2], torch.tensor(f1[3]), is_train=True, feature_type=feature_type)
    data2 = TPDataset(f2[0], f2[1], f2[2], torch.tensor(f2[3]), is_train=True, feature_type=feature_type)
    data3 = TPDataset(f3[0], f3[1], f3[2], torch.tensor(f3[3]), is_train=True, feature_type=feature_type)
    data4 = TPDataset(f4[0], f4[1], f4[2], torch.tensor(f4[3]), is_train=True, feature_type=feature_type)
    data5 = TPDataset(f5[0], f5[1], f5[2], torch.tensor(f5[3]), is_train=False, feature_type=feature_type)

    return data1, data2, data3, data4, data5

# TPTrans: A Dual-View Transformer Model for Targeting Peptide Prediction
## Description
TPTrans is a novel deep learning framework designed for macroscopic classification and microscopic cleavage site localization of targeting peptides. It synergistically integrates deep evolutionary semantics (ESM2) with structural priors (ISM) and employs multiscale convolutions to accurately parse highly heterogeneous sequences.
## Dataset Information
The datasets used in this study include:
*   **TargetP 2.0 Dataset:** Used for main model training and comparative evaluation.
*   **SignalP 6.0 Dataset:** Used for extended multispecies evaluation.
*   **Independent Data Sets (1-4):** Used for generalization and robustness evaluation.

*(Note: TargetP 2.0 and SignalP 6.0 datasets are located in the `data/` folder. Independent data sets 1 and 2 can be downloaded from [https://github.com/qianmao2001/DeepMaT], while Independent Datasets 3 and 4 should be downloaded from the UniProt database [https://www.uniprot.org/] according to the search criteria described in the paper.)*
## Code Information
The repository is structured as follows:
*   `ESM2/ESM.py` : This script leverages the **ESM-2 650M** protein language model to extract deep evolutionary semantic representations from raw FASTA sequences.
*   `ISM/ISM.py` : This script leverages the **ISM** protein language model to extract deep evolutionary semantic representations from raw FASTA sequences.
*   `composite_model.py` : Contains the core implementation of the TPTrans architecture.
*   `tool.py` : Scripts serves as the core data engine for the TPTrans framework. It handles raw data parsing, multifaceted feature encoding, dynamic tensor construction, and data augmentation.
*   `train_model.py` : The main script for training the TPTrans model.
## Requirements
To run this code, you need the following dependencies. We recommend using Python 3.9+:
*   PyTorch >= 2.1.0
*   NumPy >= 1.26.0
*   Pandas >= 2.3.0
*   Scikit-learn >= 1.6.0
*   Matplotlib >= 3.9.0
*   Seaborn >= 0.13.0

You can install all dependencies via:
```bash
pip install -r requirements.txt
```
## Usage Instructions
### 1. Data Preprocessing
Prepare your FASTA files and run the feature extraction tool:
```bash
python ESM2.py --input data/targetp_data/targetp.fasta --output data/targetp_data/ESM2
python ISM.py --input data/signalp_data/signalp_clean.fasta --output data/signalp_data/ISM
```
*(Note: ISM model files are required for full operation, just download checkpoint.pth and model.safetensors into the ism directory at https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb.
ESM model files is obtained in a similar way to the above.)*

### 2. Model Training
To train the TPTrans model from scratch using 5-fold cross-validation:
```bash
python train_model.py
```
### 3. Model Inference (Prediction)
You can predict targeting peptides and cleavage sites for new sequences using the pre-trained weights.
## Methodology
1. Feature Extraction: Protein sequences are concurrently processed by ESM2 (evolutionary semantics) and ISM (structural priors).
2. Representation Learning: A Transformer encoder captures global sequence dependencies.
3. Multiscale Perception: Parallel convolutions with kernel sizes k={3,5,9} capture local cleavage motifs.
4. Hierarchical Gating: A hard gating mechanism ensures logical consistency between macroscopic classification and microscopic localization.

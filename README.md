# TPTrans
Official PyTorch implementation of the paper **"TPTrans: Accurate Target Peptide Prediction and Cleavage Localization via Multiscale Convolution and Consistency Hierarchical Gating"**.
## 1. Clone project 
```bash
git clone https://github.com/Feir0124/TPTrans.git
cd TPTrans
```
## 2. Create conda environment using requirments.txt file
```bash
pip install -r requirements.txt
```
## 3. Data
Before running, you need to obtain the ISM features and ESM features. 

ISM model files are required for full operation, just download checkpoint.pth and model.safetensors into the ism directory at https://huggingface.co/jozhang97/ism_t33_650M_uc30pdb.
ESM model files is obtained in a similar way to the above.
```bash
python ISM/ISM.py
python ESM/ESM.py
```
## 4. Train model
```bash
python train_model.py
```

import gc
import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

import tool
from data.targetp_data.load_data import creat_data
from model.composite_model import ProteinModel


# --- Auxiliary function: calculate the cleavage site accuracy ---
def calculate_cs_accuracy(pred_seqs, true_seqs):
    correct = 0
    total = len(pred_seqs)
    if total == 0: return 0.0

    if isinstance(true_seqs, torch.Tensor):
        true_seqs_np = true_seqs.detach().cpu().numpy()
    else:
        true_seqs_np = np.array(true_seqs)

    for i in range(total):
        t_seq = true_seqs_np[i]
        t_indices = np.where(t_seq == 1)[0]
        true_cs = t_indices[-1] if len(t_indices) > 0 else -1

        p_seq = pred_seqs[i]
        if isinstance(p_seq, torch.Tensor):
            p_seq = p_seq.detach().cpu().numpy()
        else:
            p_seq = np.array(p_seq)

        p_indices = np.where(p_seq == 1)[0]
        pred_cs = p_indices[-1] if len(p_indices) > 0 else -1

        if true_cs == pred_cs:
            correct += 1

    return correct / total


# --- Core function: validation and inference ---
def validate(model, loader, device):
    model.eval()

    all_cls_preds = []
    all_cls_labels = []
    all_seq_preds = []
    all_seq_labels = []

    with torch.no_grad():
        for batch_idx, (features, tags, species) in enumerate(loader):
            features = features.to(device)
            tags = tags.to(device)
            species = species.to(device)

            cls_logits, crf_pred_path = model(features, tags=None)

            cls_preds_list = torch.argmax(cls_logits, dim=1).cpu().numpy().tolist()

            if isinstance(crf_pred_path, torch.Tensor):
                raw_seq_preds = crf_pred_path.detach().cpu().tolist()
            else:
                raw_seq_preds = crf_pred_path

            batch_size_current = len(cls_preds_list)
            crf_len = len(raw_seq_preds)
            if batch_size_current > 1 and crf_len == 1:
                if isinstance(raw_seq_preds[0], list) and len(raw_seq_preds[0]) == batch_size_current:
                    raw_seq_preds = raw_seq_preds[0]

            if len(cls_preds_list) != len(raw_seq_preds):
                min_len = min(len(cls_preds_list), len(raw_seq_preds))
                cls_preds_list = cls_preds_list[:min_len]
                raw_seq_preds = raw_seq_preds[:min_len]
                species_np = species.cpu().numpy()[:min_len]
                tags_tensor = tags[:min_len]
            else:
                species_np = species.cpu().numpy()
                tags_tensor = tags

            cleaned_seq_preds = []
            for i, pred_class in enumerate(cls_preds_list):
                seq = raw_seq_preds[i]
                if pred_class == 0:
                    if not isinstance(seq, list): seq = seq.tolist() if hasattr(seq, 'tolist') else list(seq)
                    cleaned_seq_preds.append([0] * len(seq))
                else:
                    cleaned_seq_preds.append(seq)


            all_cls_preds.extend(cls_preds_list)
            all_cls_labels.extend(species_np)
            all_seq_preds.extend(cleaned_seq_preds)

            if len(all_seq_labels) == 0:
                all_seq_labels = tags_tensor
            else:
                all_seq_labels = torch.cat((all_seq_labels, tags_tensor), dim=0)

    # --- Calculate detailed metrics ---
    min_total = min(len(all_cls_labels), len(all_cls_preds), len(all_seq_preds), len(all_seq_labels))
    y_true = np.array(all_cls_labels[:min_total])
    y_pred = np.array(all_cls_preds[:min_total])
    seq_pred = all_seq_preds[:min_total]
    seq_true = all_seq_labels[:min_total].detach().cpu().numpy()

    global_mcc = matthews_corrcoef(y_true, y_pred)
    global_cs_acc = calculate_cs_accuracy(seq_pred, seq_true)

    metrics = {
        'Global_MCC': global_mcc,
        'Global_CS_Acc': global_cs_acc
    }

    target_classes = {'Other': 0, 'SP': 1, 'MT': 2, 'CH': 3, 'TH': 4}
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

    print("\n" + "=" * 80)
    print(f"{'Class':<8} {'Prec':<8} {'Recall':<8} {'F1':<8} {'MCC':<8} {'CS_Acc':<8} {'Count':<6}")
    print("-" * 80)

    for name, idx in target_classes.items():
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

        num = (tp * tn) - (fp * fn)
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = num / den if den > 0 else 0

        class_indices = [i for i, x in enumerate(y_true) if x == idx]
        count = len(class_indices)

        if count > 0:
            sub_seq_p = [seq_pred[i] for i in class_indices]
            sub_seq_t = [seq_true[i] for i in class_indices]
            cs_acc = calculate_cs_accuracy(sub_seq_p, sub_seq_t)
        else:
            cs_acc = 0.0

        print(f"{name:<8} {p:.4f}   {r:.4f}   {f1:.4f}   {mcc:.4f}   {cs_acc:.4f}   {count:<6}")

        metrics[f'{name}_Precision'] = p
        metrics[f'{name}_Recall'] = r
        metrics[f'{name}_F1'] = f1
        metrics[f'{name}_MCC'] = mcc
        metrics[f'{name}_CS_Acc'] = cs_acc

    print("=" * 80)
    return metrics


# --- Training function: train for one epoch ---
def train_epoch(model, optimizer, loader, device, cls_criterion, scaler, scheduler):
    model.train()
    total_loss = 0

    for step, (features, tags, species) in enumerate(loader):
        features = features.to(device)
        tags = tags.to(device)
        species = species.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            cls_out, crf_loss = model(features, tags=tags)

            cls_loss = cls_criterion(cls_out, species.long())

            loss = (crf_loss * 5.0) + cls_loss

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def main():
    SEED = 42
    BATCH_SIZE = 128
    EPOCHS = 50
    LR = 0.001
    INPUT_DIM = 24
    HIDDEN_DIM = 128
    NUM_CLASSES = 5

    MODEL_SAVE_DIR = '../model_save'
    if not os.path.exists(MODEL_SAVE_DIR):
        os.makedirs(MODEL_SAVE_DIR)

    tool.set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cls_criterion = FocalLoss(gamma=2.0, alpha=None).to(device)

    print("Loading datasets...")
    datasets = list(creat_data())
    all_fold_results = []

    for fold in range(5):
        print(f'\n{"=" * 20} Fold {fold + 1} / 5 {"=" * 20}')

        train_sets = datasets[:4]
        val_set = datasets[4]

        for ds in train_sets: ds.is_train = True
        val_set.is_train = False

        # Weighted Random Sampling
        train_loaders = []
        for ds in train_sets:
            targets = ds.species
            if isinstance(targets, torch.Tensor):
                targets = targets.cpu().numpy()
            elif isinstance(targets, list):
                targets = np.array(targets)

            class_counts = np.bincount(targets)
            class_weights = 1.0 / (np.power(class_counts, 0.5) + 1e-6)
            sample_weights = class_weights[targets]

            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights).double(),
                num_samples=len(ds),
                replacement=True
            )

            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                sampler=sampler,
                num_workers=4,
                pin_memory=True
            )
            train_loaders.append(loader)

        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

        model = ProteinModel(d_model=INPUT_DIM, num_classes=NUM_CLASSES, hidden_dim=HIDDEN_DIM).to(device)

        optimizer = torch.optim.Adamax(model.parameters(), lr=LR)

        steps_per_epoch = len(train_loaders[0])
        total_training_steps = steps_per_epoch * EPOCHS

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_training_steps * 0.1),  # 10% 热身 (比如前500步)
            num_training_steps=total_training_steps
        )

        scaler = torch.cuda.amp.GradScaler()

        best_metric = -1.0
        best_epoch_metrics = {}

        for epoch in range(EPOCHS):
            train_loss = 0
            for t_loader in train_loaders:
                train_loss += train_epoch(model, optimizer, t_loader, device, cls_criterion, scaler, scheduler)

            print(f"[Epoch {epoch + 1}/{EPOCHS}] Train Loss: {train_loss / 4:.4f}")

            metrics = validate(model, val_loader, device)

            print(f" >>> [Global Monitor] MCC: {metrics['Global_MCC']:.4f} | CS_Acc: {metrics['Global_CS_Acc']:.4f}")

            if metrics['Global_MCC'] > best_metric:
                best_metric = metrics['Global_MCC']
                best_epoch_metrics = metrics.copy()
                best_epoch_metrics['Fold'] = fold + 1
                best_epoch_metrics['Best_Epoch'] = epoch + 1

                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f'best_model_fold{fold + 1}.pth'))
                print(" >>> ★★★ New Best Model Saved! ★★★")

            print("-" * 65)

        print(f"\nFold {fold + 1} Finished.")

        if 'Global_MCC' in best_epoch_metrics: del best_epoch_metrics['Global_MCC']
        if 'Global_CS_Acc' in best_epoch_metrics: del best_epoch_metrics['Global_CS_Acc']

        all_fold_results.append(best_epoch_metrics)

        if len(all_fold_results) > 0:
            df_results = pd.DataFrame(all_fold_results)
            mean_metrics = df_results.mean(numeric_only=True)
            mean_metrics['Fold'] = 'Average'
            df_final = pd.concat([df_results, pd.DataFrame([mean_metrics])], ignore_index=True)

            ordered_cols = ['Fold', 'Best_Epoch']
            for cls in ['Other', 'SP', 'MT', 'CH', 'TH']:
                ordered_cols.extend([f'{cls}_Precision', f'{cls}_Recall', f'{cls}_F1', f'{cls}_MCC', f'{cls}_CS_Acc'])

            final_cols = [c for c in ordered_cols if c in df_final.columns]
            df_final = df_final[final_cols]

            csv_path = os.path.join(MODEL_SAVE_DIR, 'evaluation_metrics.csv')

            df_final.to_csv(csv_path, index=False, float_format='%.4f')
            print(f" >> Metrics updated: {csv_path}")

        first = datasets.pop(0)
        datasets.append(first)

        del model, optimizer
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == '__main__':
    main()
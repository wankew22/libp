import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import os
from model.LiBP import LiBP
from data.MoleculeDataset import *

class EarlyStopping:
    def __init__(self, patience=20, mode='min', delta=1e-4, save_path='best_model.pth'):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self._save_best_model(model)
        elif self._is_improved(current_score):
            self.best_score = current_score
            self.counter = 0
            self._save_best_model(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _is_improved(self, current_score):
        if self.mode == 'max':
            return current_score > self.best_score + self.delta
        else:
            return current_score < self.best_score - self.delta

    def _save_best_model(self, model):
        save_dir = os.path.dirname(self.save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), self.save_path)

def train_one_epoch(model, dataloader, optimizer, criterion, device, augmentation=True):
    model.train()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    for data in dataloader:
        data = data.to(device)
        
        # Feature Augmentation
        if augmentation and hasattr(data, 'x') and data.x is not None:
            noise = torch.randn_like(data.x) * 0.005 
            data.x = data.x + noise

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        loss.backward()
        
        # --- SAFETY: Stricter Gradient Clipping ---
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
        labels = data.y.long().detach().cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs[:, 1].detach().cpu().numpy())
        total_loss += loss.item() * data.num_graphs

    # --- SAFETY: NaN Handling for Metrics ---
    all_probs = np.array(all_probs)
    if np.isnan(all_probs).any():
        print("!! WARNING: NaNs detected in output. Skipping AUC for this epoch. !!")
        auc = 0.5
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return {
        "loss": total_loss / len(dataloader.dataset),
        "Accuracy": accuracy_score(all_labels, all_preds),
        "ROC-AUC": auc
    }

def evaluate_metrics(model, dataloader, criterion, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0

    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            logits = model(data)
            loss = criterion(logits, data.y)
            probs = F.softmax(logits, dim=1)
            preds = (probs[:, 1] > 0.5).long().cpu().numpy()
            labels = data.y.long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1].cpu().numpy())
            total_loss += loss.item() * data.num_graphs

    all_probs = np.array(all_probs)
    auc = roc_auc_score(all_labels, all_probs) if not np.isnan(all_probs).any() else 0.5

    return {
        "loss": total_loss / len(dataloader.dataset),
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds, zero_division=0),
        "F1": f1_score(all_labels, all_preds, zero_division=0),
        "ROC-AUC": auc
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='/home/wangkewen/LiBP/generated_models/BBP_swa_stable.pt')
    parser.add_argument("--batch_size", type=int, default=64) 
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = MoleculeDataset(args.sdf_file, split="train")
    valid_dataset = MoleculeDataset(args.sdf_file, split="val")
    test_dataset = MoleculeDataset(args.sdf_file, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = LiBP().to(device)
    
    # --- SAFETY: Stable Weight Decay ---
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.7)
    
    # --- SAFETY: Much lower SWA Learning Rate ---
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, swa_lr=5e-5) 
    swa_start = 60 # Start later to ensure stability

    early_stopping = EarlyStopping(patience=25, mode='min', save_path=args.save_path)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_metrics(model, valid_loader, criterion, device)

        print(f"Epoch {epoch:02d} | T-Loss: {train_metrics['loss']:.4f} | V-Loss: {val_metrics['loss']:.4f} | V-Acc: {val_metrics['Accuracy']:.4f} | V-AUC: {val_metrics['ROC-AUC']:.4f}")

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])

        early_stopping(val_metrics["loss"], model)

        if early_stopping.early_stop:
            print("\n[EarlyStop] Training terminated safely.")
            break

    update_bn(train_loader, swa_model, device=device)

    # Load and Test
    if os.path.exists(args.save_path):
        model.load_state_dict(torch.load(args.save_path))
    
    test_metrics = evaluate_metrics(model, test_loader, criterion, device)
    swa_test_metrics = evaluate_metrics(swa_model, test_loader, criterion, device)

    print(f"\n--- Final Performance ---")
    print(f"Regular Model: {test_metrics}")
    print(f"SWA Model: {swa_test_metrics}")

if __name__ == "__main__":
    main()

# PYTHONPATH=. python scripts/Train_BP.py --sdf_file /Users/kewen/Desktop/LiBP/benchmark_proof/bbbp_3d_final_for_qualitative_train.sdf 
# on ssh: PYTHONPATH=. python3 scripts/Train_BP.py --sdf_file /home/wangkewen/LiBP/benchmark_proof/bbbp_3d_final_for_qualitative_train.sdf
## prev logic - Test: {'loss': 0.32584824585252337, 'Accuracy': 0.88, 'Precision': 0.8923327895595432, 'Recall': 0.9286926994906621, 'F1': 0.9101497504159733, 'ROC-AUC': 0.9355493806604469}
## early stop - Test: {'loss': 0.3170814005533854, 'Accuracy': 0.8644444444444445, 'Precision': 0.9032815198618307, 'Recall': 0.8879456706281834, 'F1': 0.8955479452054794, 'ROC-AUC': 0.9393871568247452}
# early stop with find logic: {'loss': 0.31250521209504867, 'Accuracy': 0.8711111111111111, 'Precision': 0.8796147672552167, 'Recall': 0.9303904923599321, 'F1': 0.9042904290429042, 'ROC-AUC': 0.9347632643479875}
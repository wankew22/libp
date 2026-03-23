import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
sys.path.append('/home/suqun/model/LiBP')
from model.LiBP import LiBP
from data.MoleculeDataset import *


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    for data in dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, data.y)
        loss.backward()
        optimizer.step()
        probs = F.softmax(logits, dim=1)  # 转成概率，[batch, 2]
        preds = torch.argmax(probs, dim=1).detach().cpu().numpy()  # 预测类别0或1
        labels = data.y.long().detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)
        all_probs.extend(probs[:, 1].detach().cpu().numpy())  # 取正类概率用于AUC
        total_loss += loss.item() * data.num_graphs

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)  # 用正类概率计算ROC-AUC
    loss = total_loss / len(dataloader.dataset)

    return {
        "loss": loss,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc
    }


def evaluate_metrics(model, dataloader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device)
            logits = model(data)  # shape: [batch, 2]
            loss = criterion(logits, data.y)
            probs = F.softmax(logits, dim=1)  # 转成概率，[batch, 2]
            preds = torch.argmax(probs, dim=1).cpu().numpy()  # 预测类别0或1
            labels = data.y.long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1].cpu().numpy())  # 取正类概率用于AUC
            total_loss += loss.item() * data.num_graphs

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)  # 用正类概率计算ROC-AUC
    loss = total_loss / len(dataloader.dataset)

    return {
        "loss": loss,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": auc
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_file", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='/home/suqun/model/LiBP/ckpt/BBP_nh.pt')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 使用 PyG Dataset
    train_dataset = MoleculeDataset(args.sdf_file, split="train")
    valid_dataset = MoleculeDataset(args.sdf_file, split="val")
    test_dataset = MoleculeDataset(args.sdf_file, split="test")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # 模型、损失函数和优化器
    model = LiBP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.9, verbose=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device)
        valid_metrics = evaluate_metrics(model, valid_loader, criterion, device)
        print(f"Epoch {epoch:02d} | Train Loss: {train_metrics} | Val: {valid_metrics}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        scheduler.step(valid_metrics["Accuracy"])

        if valid_metrics['loss'] < best_val_loss:
            best_val_loss = valid_metrics['loss']
            torch.save(model.state_dict(), args.save_path)

    model.load_state_dict(torch.load("best_model.pt")) # args.save_path instead?
    test_metrics = evaluate_metrics(model, test_loader, criterion, device)
    print(f"Test: {test_metrics}")


if __name__ == "__main__":
    main()

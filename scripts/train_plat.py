import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import Adam
from torch_geometric.data import Dataset, Data, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split  # 导入划分训练集和测试集的工具

from rdkit import Chem
from rdkit.Chem import AllChem
sys.path.append('/home/suqun/model/LiBP')
from plat_model.model import SubGT, GraphTransformer  # 你自己的 SubGT 文件路径


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception(f"Input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def calc_atom_features(atom, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 'Fe', 'Zn', 'Cu', 'Mn', 'Mo', 'other']
    ) + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]) + \
           [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
           one_of_k_encoding_unk(atom.GetHybridization(), [
               Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
               Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
               Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    return np.array(results)


def calc_bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(), bond.IsInRing()
    ]
    if use_chirality:
        bond_feats += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    return np.array(bond_feats).astype(int)


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # -------------------------
    # 节点特征
    # -------------------------
    x = np.array([calc_atom_features(a) for a in mol.GetAtoms()])

    # -------------------------
    # 边特征
    # -------------------------
    row, col, edge_attr = [], [], []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()

        bond_feats = calc_bond_features(bond)

        row += [a, b]
        col += [b, a]

        edge_attr.append(bond_feats)
        edge_attr.append(bond_feats)

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return Data(x=torch.tensor(x, dtype=torch.float), edge_index=edge_index, edge_attr=edge_attr)


class BBBP_Dataset(Dataset):
    def __init__(self, csv_path, cache_dir="/home/suqun/model/LiBP/dataset"):
        super().__init__()
        self.cache_dir = cache_dir
        self.csv_path = csv_path

        cache_file = os.path.join(self.cache_dir, "bbbp_graphs.pt")
        if os.path.exists(cache_file):
            print(f"Loading preprocessed data from {cache_file}")
            data = torch.load(cache_file)
            self.smiles = data["smiles"]
            self.labels = data["labels"]
            self.graphs = data["graphs"]
        else:
            df = pd.read_csv(self.csv_path)
            df = df[df["type"] == "SMILES"]

            smiles = df["sequence"].astype(str).tolist()
            labels = df["label"].astype(int).tolist()

            self.graphs = []
            self.labels = []
            self.smiles = []
            for smi, label in zip(smiles, labels):
                g = mol_to_graph(smi)
                if g is not None:
                    self.graphs.append(g)
                    self.labels.append(label)
                    self.smiles.append(smi)

            os.makedirs(self.cache_dir, exist_ok=True)
            torch.save({"smiles": self.smiles, "labels": self.labels, "graphs": self.graphs}, cache_file)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        graph = self.graphs[idx]
        graph.y = torch.tensor([self.labels[idx]], dtype=torch.float)
        return graph


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)  # SubGT forward

        # 交叉熵损失
        loss = criterion(out, batch.y.long())  # 对于交叉熵，标签需要是整数型 (long)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    preds, trues, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)

            # 使用 softmax 转换为概率
            out = torch.softmax(out, dim=-1)

            # 预测类别：选择最大概率对应的类别
            pred = torch.argmax(out, dim=-1).cpu().numpy()  # 预测类别
            probs_batch = out[:, 1].cpu().numpy()  # 取得正类别的概率（适用于二分类）

            preds.extend(pred)
            trues.extend(batch.y.cpu().numpy())
            probs.extend(probs_batch)  # 保存概率（用于 AUC 计算）

    # 转换为 PyTorch tensor
    preds = torch.tensor(preds)
    trues = torch.tensor(trues)

    # 计算 AUC (假设是二分类，probs 是正类的概率)
    auc = roc_auc_score(trues.numpy(), probs)

    # 计算 F1-Score (二分类)
    f1 = f1_score(trues.numpy(), preds.numpy())

    # 计算 MCC (Matthews Correlation Coefficient)
    mcc = matthews_corrcoef(trues.numpy(), preds.numpy())

    # 计算准确率
    acc = (preds == trues).float().mean().item()

    return auc, f1, mcc, acc


def main():
    csv_path = "/home/suqun/model/LiBP/dataset/SMILES.csv"  # 修改为你的路径
    dataset = BBBP_Dataset(csv_path)

    # 划分数据集，80% 训练集，10% 验证集，10% 测试集
    train_smiles, temp_smiles, train_labels, temp_labels = train_test_split(
        dataset.smiles, dataset.labels, test_size=0.2, random_state=42)

    # 从临时集进一步划分，50% 用于验证集，50% 用于测试集（相当于原数据集的10%）
    val_smiles, test_smiles, val_labels, test_labels = train_test_split(
        temp_smiles, temp_labels, test_size=0.5, random_state=42)

    # 创建训练集、验证集和测试集
    train_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in train_smiles]
    val_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in val_smiles]
    test_dataset = [dataset.get(i) for i in range(len(dataset)) if dataset.smiles[i] in test_smiles]

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphTransformer(
        in_channels=38,
        edge_features=6,
        num_hidden_channels=256,
    ).to(device)

    # model = SubGT(
    #     in_channels=38,
    #     edge_features=6,
    #     num_hidden_channels=256,
    #     num_layers=6
    # ).to(device)

    optimizer = Adam(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 使用 ReduceLROnPlateau 调度器，监控 val_loss 来调整学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.9, verbose=True)

    epochs = 500  # 设定训练轮数
    for epoch in range(epochs):
        # 训练过程
        loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 在测试集上评估
        auc_test, f1_test, mcc_test, acc_test = evaluate(model, test_loader, device)

        # 输出训练过程中的各项指标
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {loss}, Test AUC: {auc_test:.4f}, Test F1-Score: {f1_test:.4f}, Test MCC: {mcc_test:.4f}, "
              f"Test Accuracy: {acc_test:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")
        print("-" * 80)

        # 在每个epoch结束后调用scheduler.step(val_loss)来调整学习率
        scheduler.step(acc_test)


if __name__ == "__main__":
    main()

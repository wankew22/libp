import os
import torch
import random
from rdkit import Chem
from torch_geometric.data import InMemoryDataset, Data
from rdkit.Chem import AllChem
import numpy as np

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

class MoleculeDataset(InMemoryDataset):
    def __init__(self, sdf_file, split='train', split_ratio=(0.8, 0.1, 0.1), seed=2025,
                 transform=None, pre_transform=None):
        assert split in ['train', 'val', 'test'], "split must be one of 'train', 'val', 'test'"
        self.sdf_file = sdf_file
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        self.processed_file = os.path.splitext(sdf_file)[0] + '_pygnh.pt'
        super().__init__(os.path.dirname(self.processed_file), transform, pre_transform)

        if os.path.exists(self.processed_file):
            print(f"Loading cached PyG dataset from {self.processed_file}")
            self.full_data = torch.load(self.processed_file)
        else:
            print("Processing SDF to PyG format...")
            self.full_data = self._process_and_save()

        self.split_indices = self._get_split_indices(len(self.full_data))
        self.data_list = [self.full_data[i] for i in self.split_indices]

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]

    def _process_and_save(self):
        suppl = Chem.SDMolSupplier(self.sdf_file, removeHs=True)
        data_list = []

        for mol in suppl:
            if mol is None or mol.GetNumConformers() == 0:
                continue
            try:
                if not mol.HasProp("class"):
                    continue
                cls = mol.GetProp("class").strip().lower()
                if cls not in ("p", "n"):
                    continue
                label = 1 if cls == "p" else 0

                z = torch.tensor(np.array([calc_atom_features(a) for a in mol.GetAtoms()]), dtype=torch.float)

                # z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
                conf = mol.GetConformer()
                pos = torch.tensor([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())], dtype=torch.float)

                edge_index = []
                edge_attr = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    bond_feats = calc_bond_features(bond)

                    edge_index += [(i, j), (j, i)]

                    edge_attr.append(bond_feats)
                    edge_attr.append(bond_feats)

                # 加入自环
                num_atoms = mol.GetNumAtoms()  # 获取分子中原子的数量
                for atom_idx in range(num_atoms):
                    edge_index.append((atom_idx, atom_idx))  # 添加每个节点的自环
                    edge_attr.append(np.zeros([6]))

                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

                smiles = Chem.MolToSmiles(mol)
                data = Data(
                    z=z,
                    pos=pos,
                    edge_index=edge_index,
                    edge_attr=torch.tensor(edge_attr, dtype=torch.float32),
                    y=torch.tensor([label], dtype=torch.long),
                    name=smiles
                )
                data_list.append(data)

            except Exception as e:
                continue

        # data_obj = self.collate(data_list)
        torch.save(data_list, self.processed_file)
        print(f"Saved processed data to {self.processed_file}")
        return data_list

    def _get_split_indices(self, n):
        random.seed(self.seed)
        indices = list(range(n))
        random.shuffle(indices)
        n_train = int(self.split_ratio[0] * n)
        n_val = int(self.split_ratio[1] * n)

        if self.split == 'train':
            return indices[:n_train]
        elif self.split == 'val':
            return indices[n_train:n_train + n_val]
        else:
            return indices[n_train + n_val:]

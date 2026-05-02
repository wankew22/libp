from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter
import pandas as pd
from tqdm import tqdm

# === 设置文件路径 ===
input_csv = "/home/suqun/model/LiBP/dataset/classification.csv"        # 输入文件路径
output_sdf = "/home/suqun/model/LiBP/dataset/init_conf.sdf"  # 输出 SDF 文件路径

# === 读取 CSV 文件 ===
df = pd.read_csv(input_csv)  # 自动识别逗号分隔

# === 初始化 SDF 写入器 ===
writer = SDWriter(output_sdf)
counter = 0

# === 遍历每一行 ===
for idx, row in tqdm(df.iterrows()):
    smiles = row['SMILES']
    class_label = row['CLASS']

    # 创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"⚠️ 无法解析第 {idx} 行 SMILES：{smiles}")
        continue

    # 添加氢原子
    mol = Chem.AddHs(mol)

    # 构象嵌入（使用固定随机种子确保可复现）
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result != 0:
        print(f"⚠️ 构象生成失败：{smiles}")
        continue

    # 构象优化（可选）
    AllChem.UFFOptimizeMolecule(mol)

    # 添加分类属性
    mol.SetProp("class", str(class_label))

    counter += 1

    # 写入 SDF 文件
    writer.write(mol)

writer.close()
print(counter)
print(f"✅ 构象生成完成，输出文件：{output_sdf}")

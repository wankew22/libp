from rdkit import Chem

# 直接读你的文件
sdf_path = "/Users/kewen/Desktop/LiBP/benchmark_proof/bbbp_3d_final_for_qualitative_train.sdf"

suppl = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False, strictParsing=False)

mols = [mol for mol in suppl]
print("总共读到分子数量：", len(mols))

for i, mol in enumerate(mols):
    print(f"分子 {i} -> 是否有效：", mol is not None)
    if i >= 5:
        break
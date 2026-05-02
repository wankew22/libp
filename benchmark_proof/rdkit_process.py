# This script processes BBBP dataset for a machine learning model training pipeline. 
# Convert 2D SMILES strings of molecules into 3D molecular structures with lowest-energy conformations, 
# while preserving their BBB penetration labels for model training.
import json
from rdkit import Chem
from rdkit.Chem import AllChem

# ===================== Your Paths Here =====================
JSON_PATH = "/Users/kewen/Desktop/LiBP/media/qualitative/bbbp_complete.json"
OUT_SDF = "bbbp_3d_final_for_qualitative_train.sdf"

# ===================== Read Your JSON Structure =====================
def load_bbbp_correct(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    compounds = data["compounds"]
    samples = []

    for cid, item in compounds.items():
        smiles = item["smiles"]
        bbb_status = item["bbb_status"]
        y = 1 if bbb_status == "BBB+" else 0
        samples.append((smiles, y))

    return samples

# ===================== Generate 3D Lowest Energy Conformation =====================
def generate_3d_conformer(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        
        mol = Chem.AddHs(mol)
        
        # Generate 3D conformation
        status = AllChem.EmbedMolecule(mol)
        if status != 0:
            return None
        
        # Catch all errors from UFF optimization
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception as e:
            # If optimization fails, still return the molecule with 3D coordinates
            pass
        
        return mol

    except Exception as e:
        return None

# ===================== Save SDF + Write y Labels =====================
def save_sdf_with_y(mol_list, y_list, output_path):
    writer = Chem.SDWriter(output_path)
    for mol, y in zip(mol_list, y_list):
        if mol:
            mol.SetProp("y", str(y))
            writer.write(mol)
    writer.close()

# ===================== Main Workflow =====================
if __name__ == "__main__":
    print("Reading BBBP data...")
    samples = load_bbbp_correct(JSON_PATH)

    mols_3d = []
    labels = []
    total = len(samples)
    print(f"Total {total} molecules, generating 3D conformations...")

    for i, (smi, y) in enumerate(samples):
        mol = generate_3d_conformer(smi)
        if mol:
            mols_3d.append(mol)
            labels.append(y)
        else:
            print(f"[{i+1}/{total}] Skipping unprocessable: {smi}")
        
        # Progress bar
        if (i+1) % 500 == 0:
            print(f"Progress: {i+1}/{total} | Successful: {len(mols_3d)}")

    print(f"\n Processing complete!")
    print(f"Total molecules: {total}")
    print(f"Successfully generated 3D: {len(mols_3d)}")

    # Writes all successfully generated 3D molecules to an SDF file
    # Attaches the binary label (y) as a molecule property for training
    save_sdf_with_y(mols_3d, labels, OUT_SDF)

    print(f"\nFile saved: {OUT_SDF}")
    print("\nTraining command:")
    print("python scripts/Train_BP.py --sdf_file", OUT_SDF)
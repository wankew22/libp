from rdkit import Chem

pdb_path = "AMI_struture/results/100_model.pdb"
mol = Chem.MolFromPDBFile(pdb_path, sanitize=True, proximityBonding=False)
if mol:
    print(f"Success! Atoms: {mol.GetNumAtoms()}")
else:
    print("RDKit failed to parse the PDB file.")

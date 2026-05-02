from rdkit import Chem

suppl = Chem.SDMolSupplier("benchmark_proof/alphafold3_combined.sdf", removeHs=False)
for i, mol in enumerate(suppl):
    if mol is None:
        print(f"Molecule {i} is None")
        continue
    
    print(f"Molecule {i}:")
    print(f"  Num Atoms: {mol.GetNumAtoms()}")
    print(f"  Num Conformers: {mol.GetNumConformers()}")
    if mol.GetNumConformers() > 0:
        conf = mol.GetConformer()
        print(f"  Is 3D: {conf.Is3D()}")
        print(f"  First atom pos: {conf.GetAtomPosition(0).x}, {conf.GetAtomPosition(0).y}, {conf.GetAtomPosition(0).z}")
    print(f"  Has 'y' prop: {mol.HasProp('y')}")
    if mol.HasProp('y'):
        print(f"  'y' prop: {mol.GetProp('y')}")
    break

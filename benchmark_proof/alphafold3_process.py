import os
import glob
import pandas as pd
from rdkit import Chem
from Bio.PDB import MMCIFParser, PDBIO

def get_label_from_filename(cif_path, df):
    """Extract index from filename and get corresponding label from dataframe."""
    filename = os.path.basename(cif_path)
    idx = int(filename.split('_')[0])
    # The file corresponds to index-1 in the dataframe
    return df.iloc[idx - 1]['label']

def main():
    # Load the dataset
    # Find the project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "dataset", "AMI_filter.csv")
    df = pd.read_csv(csv_path)

    # Target directory to process
    input_dir = os.path.join(base_dir, "AMI_struture", "results/")
    cif_files = glob.glob(f"{input_dir}*.cif")
    if not cif_files:
        print(f"Warning: No .cif files found in {input_dir}")
        if not os.path.exists(input_dir):
            print(f"   Directory doesn't exist: {input_dir}")
        raise FileNotFoundError(f"Cannot find any .cif files in {input_dir}")

    # Sort files by index to ensure consistent ordering in the combined SDF
    def get_index(path):
        filename = os.path.basename(path)
        try:
            return int(filename.split('_')[0])
        except ValueError:
            return float('inf')
            
    cif_files.sort(key=get_index)

    combined_sdf_path = os.path.join(base_dir, "benchmark_proof", "alphafold3_combined.sdf")
    combined_writer = Chem.SDWriter(combined_sdf_path)

    # Initialize BioPython parsers
    parser = MMCIFParser()
    io = PDBIO()

    success_count = 0
    for cif_path in cif_files:
        try:
            label = get_label_from_filename(cif_path, df)
        except Exception as e:
            print(f"⚠️ Failed to get label for {cif_path}: {e}")
            continue

        # 1. Convert CIF to PDB using BioPython
        try:
            structure = parser.get_structure("temp", cif_path)
            io.set_structure(structure)
            temp_pdb_path = cif_path.replace(".cif", ".pdb")
            io.save(temp_pdb_path)
        except Exception as e:
            print(f"⚠️ BioPython failed to parse CIF {cif_path}: {e}")
            continue

        # 2. Read PDB into RDKit (proximityBonding=False is crucial for proteins to avoid valence errors)
        mol = Chem.MolFromPDBFile(temp_pdb_path, sanitize=True, proximityBonding=False)
        
        # Clean up temporary PDB file
        if os.path.exists(temp_pdb_path):
            os.remove(temp_pdb_path)

        if mol:
            # Add hydrogens and generate their 3D coordinates based on the existing heavy atom 3D conformation
            mol = Chem.AddHs(mol, addCoords=True)
            
            # Add the corresponding label as 'y' property to the molecule
            mol.SetProp("y", str(label))
            
            # Write to combined SDF
            combined_writer.write(mol)
            
            print(f"✅ Added to combined SDF: {os.path.basename(cif_path)} (label: {label})")
            success_count += 1
        else:
            print(f"⚠️ Conversion failed: {cif_path}, RDKit could not parse the converted PDB file")

    combined_writer.close()
    print(f"\nProcessing complete! Successfully converted {success_count} files.")
    print(f"Combined SDF saved to: {combined_sdf_path}")

if __name__ == "__main__":
    main()
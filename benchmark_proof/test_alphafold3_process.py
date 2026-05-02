import os
import glob
import pandas as pd
from alphafold3_process import get_label_from_filename

def test_real_files_label_extraction():
    """Test label extraction on the actual CIF files and dataset."""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "dataset", "AMI_filter.csv")
    input_dir = os.path.join(base_dir, "AMI_struture", "results/")
    
    # Assert that the dataset exists
    assert os.path.exists(csv_path), f"Dataset not found at {csv_path}"
    
    # Load the real dataset
    df = pd.read_csv(csv_path)
    assert not df.empty, "Dataset is empty"
    assert 'label' in df.columns, "Dataset does not contain a 'label' column"
    
    # Assert that the input directory exists
    assert os.path.exists(input_dir), f"Results directory not found at {input_dir}"
    
    # Get all real CIF files
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    assert len(cif_files) > 0, f"No .cif files found in {input_dir}"
    
    # Sort files by the index in their filename to print in increasing order
    def get_index(path):
        filename = os.path.basename(path)
        try:
            return int(filename.split('_')[0])
        except ValueError:
            return float('inf')  # Put files that don't match the pattern at the end
            
    cif_files.sort(key=get_index)
    
    success_count = 0
    failed_files = []
    
    # Test each real file
    print("\n--- Label Extraction Results ---")
    for cif_path in cif_files:
        filename = os.path.basename(cif_path)
        try:
            label = get_label_from_filename(cif_path, df)
            # Ensure we actually got a value (not None or NaN if pandas returns that)
            assert pd.notna(label), f"Label is NaN for {cif_path}"
            
            # Print out the match to verify it's correct
            idx = int(filename.split('_')[0])
            print(f"File: {filename:20s} | Extracted Index: {idx:4d} | Label: {label}")
            
            success_count += 1
        except Exception as e:
            failed_files.append((cif_path, str(e)))
            
    print("--------------------------------\n")
            
    # Print out any failures for debugging
    if failed_files:
        print(f"\nFailed to extract labels for {len(failed_files)} files:")
        for path, error in failed_files[:10]:  # Show first 10 failures
            print(f"  {os.path.basename(path)}: {error}")
            
    # Assert that ALL files successfully found their corresponding labels
    assert len(failed_files) == 0, f"Failed to find labels for {len(failed_files)} out of {len(cif_files)} files"
    print(f"\nSuccessfully verified labels for all {success_count} real CIF files.")

if __name__ == "__main__":
    test_real_files_label_extraction()

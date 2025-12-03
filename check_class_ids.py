"""
Script to check what class IDs are used in each folder
This helps verify the class mapping in data_oops.yaml
"""
import os
from collections import Counter
from pathlib import Path

def check_class_ids_in_folder(folder_path):
    """Check all class IDs used in label files within a folder"""
    class_ids = []
    txt_files = list(Path(folder_path).glob("*.txt"))
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Non-empty line
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            class_ids.append(int(parts[0]))
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
    
    return Counter(class_ids)

def main():
    base_dir = Path("Dataset")
    splits = ["train", "valid", "test"]
    
    class_mapping = {
        0: "charging-cable",
        1: "earphones",
        2: "glasses",
        3: "keys",
        4: "paper",
        5: "pen",
        6: "student-id",
        7: "wallet",
        8: "watch",
        9: "ipad"
    }
    
    print("=" * 60)
    print("Checking class IDs used in each folder")
    print("=" * 60)
    
    for split in splits:
        split_dir = base_dir / split
        if not split_dir.exists():
            continue
            
        print(f"\n[{split.upper()}]")
        print("-" * 60)
        
        # Get all class folders
        class_folders = [d for d in split_dir.iterdir() if d.is_dir()]
        class_folders.sort()
        
        for class_folder in class_folders:
            folder_name = class_folder.name
            class_ids = check_class_ids_in_folder(class_folder)
            
            if class_ids:
                unique_ids = sorted(class_ids.keys())
                id_counts = ", ".join([f"ID {k}: {v} annotations" for k, v in sorted(class_ids.items())])
                
                print(f"\n  {folder_name}/")
                print(f"    Used class IDs: {unique_ids}")
                print(f"    Breakdown: {id_counts}")
                
                # Check if folder name matches expected class
                if folder_name in ["charging-cable", "earphones", "glasses", "keys", "paper", 
                                  "pen", "student-id", "wallet", "watch", "ipad"]:
                    # Find which class ID this folder should have
                    expected_id = None
                    for cid, cname in class_mapping.items():
                        if cname == folder_name:
                            expected_id = cid
                            break
                    
                    if expected_id is not None:
                        if expected_id in unique_ids:
                            if len(unique_ids) == 1:
                                print(f"    ✓ Matches expected class ID {expected_id}")
                            else:
                                print(f"    ⚠ Contains expected ID {expected_id} but also has other IDs: {unique_ids}")
                        else:
                            print(f"    ✗ Expected class ID {expected_id} but found: {unique_ids}")
                elif folder_name == "multiple":
                    print(f"    ℹ 'multiple' folder - contains images with multiple classes")
                    if max(unique_ids) > 9 or min(unique_ids) < 0:
                        print(f"    ⚠ Contains class IDs outside range 0-9!")
            else:
                print(f"\n  {folder_name}/: No annotations found")

if __name__ == "__main__":
    main()


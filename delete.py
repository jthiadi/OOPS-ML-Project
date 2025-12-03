import os

def is_txt_file_empty(txt_path):
    """
    Check if a txt file is empty (0 bytes or contains only whitespace).
    
    Args:
        txt_path: Path to the txt file
        
    Returns:
        True if file is empty, False otherwise
    """
    try:
        # Check file size
        if os.path.getsize(txt_path) == 0:
            return True
        
        # Check if file contains only whitespace
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            return len(content) == 0
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return False

def find_corresponding_image(txt_path):
    """
    Find the corresponding image file for a txt file.
    Tries common image extensions: .jpg, .jpeg, .png
    
    Args:
        txt_path: Path to the txt file
        
    Returns:
        Path to the image file if found, None otherwise
    """
    base_name = os.path.splitext(txt_path)[0]
    image_extensions = ['.jpg', '.jpeg', '.png']
    
    for ext in image_extensions:
        image_path = base_name + ext
        if os.path.exists(image_path):
            return image_path
    
    return None

def delete_empty_txt_and_image(root_dirs):
    """
    Delete txt files that are empty and their corresponding image files.
    
    Args:
        root_dirs: List of directory paths to search
    """
    deleted_txt_count = 0
    deleted_image_count = 0
    deleted_files = []
    
    for root_dir in root_dirs:
        if not os.path.exists(root_dir):
            print(f"Warning: Directory '{root_dir}' does not exist. Skipping...")
            continue
            
        print(f"\nScanning '{root_dir}'...")
        
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for filename in filenames:
                # Only process txt files
                if not filename.lower().endswith('.txt'):
                    continue
                
                txt_path = os.path.join(dirpath, filename)
                
                try:
                    # Check if txt file is empty
                    if is_txt_file_empty(txt_path):
                        # Delete the txt file
                        os.remove(txt_path)
                        deleted_txt_count += 1
                        deleted_files.append(('txt', txt_path))
                        print(f"Deleted empty txt: {txt_path}")
                        
                        # Find and delete corresponding image file
                        image_path = find_corresponding_image(txt_path)
                        if image_path:
                            os.remove(image_path)
                            deleted_image_count += 1
                            deleted_files.append(('image', image_path))
                            print(f"Deleted corresponding image: {image_path}")
                        else:
                            print(f"Warning: No corresponding image found for {txt_path}")
                            
                except Exception as e:
                    print(f"Error processing {txt_path}: {e}")
    
    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Empty txt files deleted: {deleted_txt_count}")
    print(f"  Corresponding image files deleted: {deleted_image_count}")
    print(f"  Total files deleted: {deleted_txt_count + deleted_image_count}")
    
    if deleted_files:
        print(f"\nDeleted files:")
        for file_type, file_path in deleted_files:
            print(f"  [{file_type}] {file_path}")

if __name__ == "__main__":
    # Define the directories to scan
    base_dir = "Dataset"
    directories = [
        os.path.join(base_dir, "test"),
        os.path.join(base_dir, "train"),
        os.path.join(base_dir, "valid")
    ]
    
    print("Starting to scan for empty txt files and their corresponding images...")
    delete_empty_txt_and_image(directories)
    print("\nDone!")

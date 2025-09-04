import os
import shutil
from pathlib import Path

def split_train_data(source_dir, batch_size=10000):
    """
    Split PNG files in the train directory into batches of specified size.
    
    Args:
        source_dir (str): Path to the train directory containing PNG files
        batch_size (int): Number of files per batch (default: 10000)
    """
    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"Source directory {source_dir} does not exist!")
        return
    
    # Get all PNG files and sort them numerically
    png_files = [f for f in os.listdir(source_path) if f.endswith('.png')]
    
    # Sort files numerically (0.png, 1.png, ..., 10.png, ..., 100.png, etc.)
    png_files.sort(key=lambda x: int(x.split('.')[0]))
    
    total_files = len(png_files)
    print(f"Found {total_files} PNG files")
    
    # Calculate number of batches
    num_batches = (total_files + batch_size - 1) // batch_size
    print(f"Will create {num_batches} batches")
    
    # Create batch directories and move files
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_files)
        
        # Create batch directory
        batch_dir = source_path / f"batch_{batch_num + 1:02d}"
        batch_dir.mkdir(exist_ok=True)
        
        print(f"Processing batch {batch_num + 1}/{num_batches}: files {start_idx} to {end_idx - 1}")
        
        # Move files to batch directory
        for i in range(start_idx, end_idx):
            src_file = source_path / png_files[i]
            dst_file = batch_dir / png_files[i]
            
            try:
                shutil.move(str(src_file), str(dst_file))
            except Exception as e:
                print(f"Error moving {src_file}: {e}")
        
        print(f"Batch {batch_num + 1} completed: {end_idx - start_idx} files moved to {batch_dir}")
    
    print(f"\nAll files have been organized into {num_batches} batches!")

if __name__ == "__main__":
    train_dir = r"Z:\Data\1_Work\Zeta\20250903_Training\train"
    split_train_data(train_dir, batch_size=10000)
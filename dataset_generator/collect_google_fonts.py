import os
import shutil

# - Configuration -

# Set the path to the directory containing the folder.
SOURCE_ROOT_DIR = '/home/bernardoribeiro/Documents/GitHub/DL-Project/dataset_generator'

# The specific subdirectory structure to start searching within
SEARCH_START_DIR = os.path.join(SOURCE_ROOT_DIR, 'ofl')

# The target directory where all .ttf files will be copied
TARGET_DIR = '/home/bernardoribeiro/Documents/GitHub/DL-Project/dataset_generator/fonts'

# - Script Logic -

def collect_ttf_fonts(start_dir, target_dir):
    """
    Recursively finds .ttf files in start_dir and its subdirectories
    and copies them to target_dir.
    """
    copied_count = 0
    skipped_count = 0
    target_dir = os.path.abspath(target_dir) # Get absolute path for clarity

    # Create the target directory if it doesn't exist
    try:
        os.makedirs(target_dir, exist_ok=True)
        print(f"Target directory '{target_dir}' ensured.")
    except OSError as e:
        print(f"Error creating target directory '{target_dir}': {e}")
        return

    print(f"Starting search in: {os.path.abspath(start_dir)}")

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(start_dir):
        for filename in filenames:
            # Check if the file ends with .ttf (case-insensitive)
            if filename.lower().endswith('.ttf'):
                source_file_path = os.path.join(dirpath, filename)
                dest_file_path = os.path.join(target_dir, filename)

                if os.path.exists(dest_file_path):
                     print(f"Warning: File '{filename}' already exists in target. Overwriting.")

                try:
                    shutil.copy2(source_file_path, dest_file_path)
                    copied_count += 1
                except Exception as e:
                    print(f"Error copying '{source_file_path}': {e}")
                    skipped_count += 1

    print(f"\nFinished.")
    print(f"Successfully copied {copied_count} .ttf files.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} files (due to errors or existing files if skipping).")
    if copied_count == 0 and skipped_count == 0:
        print(f"No .ttf files found in '{os.path.abspath(start_dir)}' or its subdirectories.")

# - Run the script -
if __name__ == "__main__":
    # Verify that the search start directory exists
    if not os.path.isdir(SEARCH_START_DIR):
        print(f"Error: The specified search start directory does not exist: '{os.path.abspath(SEARCH_START_DIR)}'")
        print(f"Please check the SOURCE_ROOT_DIR and ensure the '{os.path.basename(SEARCH_START_DIR)}' directory is inside it.")
    else:
        collect_ttf_fonts(SEARCH_START_DIR, TARGET_DIR)
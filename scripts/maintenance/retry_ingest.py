import os
import shutil
import subprocess
import re
from pathlib import Path
import difflib

# Configuration
FAILED_LIST_PATH = r"c:\Users\G\Desktop\Alexandria\failed_papers_list.txt"
LIBRARY_PATH = r"c:\Users\G\Desktop\Alexandria\data\library\arxiv"
TEMP_INGEST_PATH = r"c:\Users\G\Desktop\Alexandria\data\temp_retry_ingest"
INGEST_SCRIPT_PATH = r"c:\Users\G\Desktop\Alexandria\scripts\ingest.py"

def normalize(text):
    """Normalize text for comparison: lowercase, remove non-alphanumeric."""
    return re.sub(r'[^a-z0-9]', '', text.lower())

def find_file(title, file_map):
    """Find the file corresponding to the title."""
    # Attempt 1: Exact match with underscores
    candidate = title.replace(" ", "_") + ".pdf"
    if candidate in file_map:
        return file_map[candidate]
    
    # Attempt 2: Normalized match
    norm_title = normalize(title)
    for filename, filepath in file_map.items():
        if normalize(filename) == norm_title:
             return filepath
        # check if norm_title is a substring of filename (for truncated filenames)
        if norm_title in normalize(filename):
             return filepath

    # Attempt 3: Fuzzy match (if needed, but simple normalization covers most)
    # This might be slow if we have many files, but let's try strict first.
    return None

def main():
    print("Reading failed papers list...")
    try:
        with open(FAILED_LIST_PATH, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except UnicodeDecodeError:
        with open(FAILED_LIST_PATH, 'r', encoding='utf-16') as f:
             lines = f.readlines()

    titles = []
    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            titles.append(line[2:].strip())

    print(f"Found {len(titles)} papers to retry.")

    print("Indexing library files...")
    library_path = Path(LIBRARY_PATH)
    files = list(library_path.glob("*.pdf"))
    file_map = {f.name: f for f in files}
    print(f"Index complete. Found {len(files)} files in library.")

    # Create temp directory
    if os.path.exists(TEMP_INGEST_PATH):
        shutil.rmtree(TEMP_INGEST_PATH)
    os.makedirs(TEMP_INGEST_PATH)

    found_count = 0
    not_found = []

    print("Matching files...")
    for title in titles:
        filepath = find_file(title, file_map)
        if filepath:
            # Copy to temp dir
            shutil.copy2(filepath, os.path.join(TEMP_INGEST_PATH, filepath.name))
            found_count += 1
        else:
            not_found.append(title)
            print(f"WARNING: Could not find file for '{title}'")

    print(f"Prepared {found_count} files for ingestion.")
    if not_found:
        print(f"Could not find {len(not_found)} files.")

    if found_count == 0:
        print("No files to ingest. Exiting.")
        return

    print("Starting ingestion...")
    result = subprocess.run(["python", INGEST_SCRIPT_PATH, TEMP_INGEST_PATH, "--workers", "4"], capture_output=True, text=True)
    
    print("Ingestion Output:")
    print(result.stdout)
    if result.stderr:
        print("Ingestion Errors:")
        print(result.stderr)

    print("Cleaning up...")
    # shutil.rmtree(TEMP_INGEST_PATH) # Keep it for debugging for now or user request? The user said "attempt to ingest". Cleaning up is properly hygienic.
    # Actually, I'll keep it if it fails, but the prompt implies a one-off. I'll clean it.
    try:
       shutil.rmtree(TEMP_INGEST_PATH)
    except Exception as e:
       print(f"Error cleaning up: {e}")

    print("Done.")

if __name__ == "__main__":
    main()

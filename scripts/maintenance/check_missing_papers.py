import os
import re
from pathlib import Path

# Configuration
FAILED_LIST_PATH = r"c:\Users\G\Desktop\Alexandria\failed_papers_list.txt"
LIBRARY_PATH = r"c:\Users\G\Desktop\Alexandria\data\library\arxiv"

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
    for filename in file_map:
        if normalize(filename) == norm_title:
             return file_map[filename]
        # check if norm_title is a substring of filename (for truncated filenames or partial matches)
        if norm_title in normalize(filename):
             return file_map[filename]

    return None

def main():
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

    library_path = Path(LIBRARY_PATH)
    files = list(library_path.glob("*.pdf"))
    file_map = {f.name: f for f in files}

    print(f"Checking {len(titles)} papers against {len(files)} files in library...\n")
    
    missing_count = 0
    print("--- PAPERS NOT FOUND ---")
    for title in titles:
        filepath = find_file(title, file_map)
        if not filepath:
            print(f"- {title}")
            missing_count += 1
            
    print(f"\nTotal Missing: {missing_count}")

if __name__ == "__main__":
    main()

import os
from pathlib import Path

root = Path('.')

# Count Python files
py_files = list(root.rglob('*.py'))
total_files = len(py_files)

# Calculate total size and lines
total_size = 0
total_lines = 0

for f in py_files:
    try:
        total_size += f.stat().st_size
        with open(f, 'r', encoding='utf-8', errors='ignore') as file:
            total_lines += len(file.readlines())
    except:
        pass

print(f'Total Python files: {total_files}')
print(f'Total lines of code: {total_lines:,}')
print(f'Total size: {total_size/1024/1024:.2f} MB')

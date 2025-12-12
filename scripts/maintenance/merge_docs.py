import os

def merge_docs(source_dir, output_file):
    print(f"Scanning {source_dir}...")
    
    all_content = []
    
    # Walk tree to find all md files
    md_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".md"):
                md_files.append(os.path.join(root, file))
    
    # Sort for deterministic output
    md_files.sort()
    
    print(f"Found {len(md_files)} markdown files.")
    
    header = """# 📚 TODA CORE DOCUMENTATION
> Auto-generated compilation of all files in docs/core/

---

"""
    all_content.append(header)

    for file_path in md_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Create a separator
            relative_path = os.path.relpath(file_path, ".")
            file_header = f"\n\n<!-- ================================================== -->\n"
            file_header += f"# 📄 SOURCE: {relative_path}\n"
            file_header += f"<!-- ================================================== -->\n\n"
            
            all_content.append(file_header)
            all_content.append(content)
            print(f"Merged: {relative_path}")
            
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # Write output
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(all_content))
        print(f"✅ Successfully created {output_file} with {len(all_content)} sections.")
    except Exception as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    merge_docs("docs/core", "toda_core.md")

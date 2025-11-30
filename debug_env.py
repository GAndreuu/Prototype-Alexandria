import os
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY")
print(f"Loaded via dotenv: {key is not None}")
if key:
    print(f"Key length: {len(key)}")
    print(f"Key start: {key[:4]}...")

# Manual check
try:
    with open(".env", "r") as f:
        content = f.read()
        print(f"File size: {len(content)}")
        if "GOOGLE_API_KEY" in content:
            print("GOOGLE_API_KEY found in file text")
            for line in content.splitlines():
                if line.startswith("GOOGLE_API_KEY="):
                    val = line.split("=")[1].strip()
                    print(f"Manual parse: {val[:4]}... (len={len(val)})")
        else:
            print("GOOGLE_API_KEY NOT found in file text")
except Exception as e:
    print(f"Error reading file: {e}")

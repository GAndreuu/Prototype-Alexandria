import arxiv
import os
import time
import re
import random

# Configuration
SAVE_DIR = r"c:\Users\G\Desktop\Alexandria\data\library\recovered_knowledge"

# Topics derived from lost papers
QUERIES = [
    "Abductive Reasoning Large Language Models",
    "Neuro-Symbolic AI Reasoning",
    "Vector-Symbolic Architectures",
    "Spiking Neural Networks Algorithms",
    "Geometric Transformers Field Theory",
    "Quantized 3D Object Detection",
    "Medical Dialogue Reasoning",
    "Sparse Orthogonal Dictionary Learning",
    "Neural Control Barrier Certificates",
    "Monotone Neural Control",
    "Visual Abstract Reasoning",
    "Rationality and Knowledge AI",
    "Compressive Sensing Reconstruction",
    "Vector Quantized Variational Autoencoder"
]

def normalize_title(title):
    return re.sub(r'[^a-zA-Z0-9_\-]', '_', title)[:150]

def resilient_search(query, max_results=5):
    """Executes search with exponential backoff for 429s."""
    delay = 10
    attempts = 0
    max_attempts = 5
    
    print(f"\n[Search] '{query}'")
    
    while attempts < max_attempts:
        client = arxiv.Client(
            page_size=max_results,
            delay_seconds=3.0,
            num_retries=1 
        )
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        try:
            # We must iterate to actually trigger the request
            results = list(client.results(search))
            return results
        except Exception as e:
            if "429" in str(e) or "HTTP Error 429" in str(e):
                wait_time = delay * (2 ** attempts) + random.uniform(1, 5)
                print(f"  !! Rate limited (429). Waiting {wait_time:.1f}s before retry {attempts+1}/{max_attempts}...")
                time.sleep(wait_time)
                attempts += 1
            else:
                print(f"  !! Unexpected error: {e}")
                return []
    
    print("  !! Max retries reached. Skipping.")
    return []

def resilient_download(result, save_dir):
    """Downloads pdf with backoff."""
    title = result.title
    filename = normalize_title(title) + ".pdf"
    filepath = os.path.join(save_dir, filename)

    if os.path.exists(filepath):
        print(f"  -> Skipping (exists): {title[:30]}...")
        return

    print(f"  -> Downloading: {title[:50]}...")
    
    delay = 5
    attempts = 0
    while attempts < 3:
        try:
            result.download_pdf(dirpath=save_dir, filename=filename)
            print("     Success.")
            return
        except Exception as e:
            wait_time = delay * (2 ** attempts)
            print(f"     Download failed: {e}. Retry in {wait_time}s...")
            time.sleep(wait_time)
            attempts += 1

def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    print(f"Starting resilient mass recovery for {len(QUERIES)} topics.")
    print("This may take time due to API rate limits.")

    total_downloaded = 0
    
    for query in QUERIES:
        results = resilient_search(query)
        if not results:
            continue
            
        for r in results:
            resilient_download(r, SAVE_DIR)
            total_downloaded += 1
            # Respectful delay between downloads
            time.sleep(random.uniform(5, 10))
        
        # Respectful delay between queries
        time.sleep(random.uniform(10, 20))            

    print(f"\nRecovery Complete. Downloaded approximately {total_downloaded} papers.")

if __name__ == "__main__":
    main()

"""
Script to convert wiki-trained model weights to MonolithV13 format.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

def convert_wiki_to_v13(wiki_path, output_path):
    """
    Convert wiki-trained model weights to MonolithV13 format.
    
    The wiki model has a slightly different structure but compatible dimensions.
    We just need to remap the layer names and extract the codebooks.
    """
    print(f"Loading wiki weights from {wiki_path}...")
    wiki_state = torch.load(wiki_path, map_location='cpu', weights_only=False)
    
    print("\nWiki model structure:")
    for k, v in sorted(wiki_state.items()):
        print(f"  {k}: {v.shape}")
    
    # Create new state dict for MonolithV13
    v13_state = {}
    
    # Map encoder layers (wiki has 5 layers: 0-4)
    # MonolithV13 also has 5 layers: 0-4
    v13_state['encoder.0.weight'] = wiki_state['encoder.0.weight']
    v13_state['encoder.0.bias'] = wiki_state['encoder.0.bias']
    v13_state['encoder.1.weight'] = wiki_state['encoder.1.weight']
    v13_state['encoder.1.bias'] = wiki_state['encoder.1.bias']
    # Layer 2 is GELU (no params)
    v13_state['encoder.3.weight'] = wiki_state['encoder.3.weight']  
    v13_state['encoder.3.bias'] = wiki_state['encoder.3.bias']
    v13_state['encoder.4.weight'] = wiki_state['encoder.4.weight']
    v13_state['encoder.4.bias'] = wiki_state['encoder.4.bias']
    
    # Map quantizer codebooks
    v13_state['quantizer.codebooks'] = wiki_state['quantizer.codebooks']
    
    # Map decoder layers
    v13_state['decoder.0.weight'] = wiki_state['decoder.0.weight']
    v13_state['decoder.0.bias'] = wiki_state['decoder.0.bias']
    v13_state['decoder.1.weight'] = wiki_state['decoder.1.weight']
    v13_state['decoder.1.bias'] = wiki_state['decoder.1.bias']
    # Layer 2 is GELU (no params)
    v13_state['decoder.3.weight'] = wiki_state['decoder.3.weight']
   v13_state['decoder.3.bias'] = wiki_state['decoder.3.bias']
    
    print(f"\n✅ Converted {len(wiki_state)} → {len(v13_state)} params")
    
    # Save
    torch.save(v13_state, output_path)
    print(f"✅ Saved to {output_path}")
    
    return v13_state

if __name__ == "__main__":
    wiki_path = "data/monolith_v13_wiki_trained.pth"
    output_path = "data/monolith_v13_wiki_converted.pth"
    
    convert_wiki_to_v13(wiki_path, output_path)

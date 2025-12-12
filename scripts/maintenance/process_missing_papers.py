"""
Script para processar papers faltantes.
Compara PDFs na pasta com sources no LanceDB e processa apenas os faltantes.
"""
import os
import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.memory.semantic_memory import SemanticFileSystem
from core.topology.topology_engine import TopologyEngine
import lancedb

def get_processed_sources():
    """Retorna set de sources já processados."""
    db = lancedb.connect('data/lancedb_store')
    table = db.open_table('semantic_memory')
    df = table.to_pandas()
    return set(df['source'].unique())

def get_all_pdfs():
    """Retorna todos os PDFs na biblioteca."""
    pdf_dir = Path('data/library/arxiv')
    return list(pdf_dir.glob('*.pdf'))

def main():
    print("🔍 Identificando papers faltantes...")
    
    # Get já processados
    processed = get_processed_sources()
    print(f"✅ Papers já processados: {len(processed)}")
    
    # Get todos os PDFs
    all_pdfs = get_all_pdfs()
    print(f"📚 Total de PDFs: {len(all_pdfs)}")
    
    # Identificar faltantes
    missing = []
    for pdf in all_pdfs:
        # Verificar se o nome do arquivo está nos sources
        pdf_name = pdf.name
        if pdf_name not in processed and str(pdf) not in processed:
            missing.append(pdf)
    
    print(f"❌ Papers faltantes: {len(missing)}")
    
    if not missing:
        print("✨ Todos os papers já foram processados!")
        return
    
    # Processar faltantes
    print("\n🚀 Iniciando processamento dos faltantes...")
    
    topology = TopologyEngine()
    sfs = SemanticFileSystem(topology)
    
    success = 0
    failed = 0
    
    for i, pdf in enumerate(missing):
        try:
            print(f"[{i+1}/{len(missing)}] Processando: {pdf.name[:50]}...")
            chunks = sfs.index_file(str(pdf), doc_type="SCI")
            if chunks > 0:
                success += 1
                print(f"  ✅ {chunks} chunks indexados")
            else:
                failed += 1
                print(f"  ⚠️ 0 chunks (PDF vazio ou erro)")
        except Exception as e:
            print(f"  ❌ Erro: {str(e)[:50]}")
            failed += 1
        
        # Progress a cada 100
        if (i + 1) % 100 == 0:
            print(f"  📊 Progresso: {success} OK, {failed} falhas")
    
    print(f"\n✅ Concluído: {success} processados, {failed} falhas")

if __name__ == "__main__":
    main()

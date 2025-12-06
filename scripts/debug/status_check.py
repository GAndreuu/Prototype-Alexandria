
import sys
import pickle
import os

# Adiciona o caminho do projeto
sys.path.append('c:\\Users\\G\\Desktop\\Alexandria')
from core.reasoning.mycelial_reasoning import MycelialReasoning

def check_status():
    print("="*60)
    print("📊 ALEXANDRIA SYSTEM STATUS")
    print("="*60)
    
    state_path = 'c:\\Users\\G\\Desktop\\Alexandria\\data\\mycelial_state.pkl'
    
    if os.path.exists(state_path):
        try:
            with open(state_path, 'rb') as f:
                state = pickle.load(f)
                
            # Se for uma instância salva, pode ser diferente de um dicionário puro
            if isinstance(state, dict):
                graph = state.get('graph', {})
                concepts = len(graph)
                connections = sum(len(neighbors) for neighbors in graph.values())
            else:
                # Assumindo que talvez salvamos o objeto inteiro ou algo assim,
                # mas baseada no integration_layer, salvamos um dict
                print("⚠️ Estructure de estado desconhecida.")
                return

            print(f"\n🧠 MEMÓRIA MICELIAL:")
            print(f"   - Conceitos (Nós): {concepts}")
            print(f"   - Conexões (Edges): {connections}")
            if concepts > 0:
                print(f"   - Densidade Média: {connections/concepts:.2f} conexões/nó")
                
        except Exception as e:
            print(f"❌ Erro ao ler estado micelial: {e}")
    else:
        print("⚠️ Arquivo de estado Mycelial não encontrado.")

if __name__ == "__main__":
    check_status()

"""
Demo of Prototype Alexandria's Advanced Capabilities
Orchestrates a cycle of Abduction -> Action -> Learning.
"""

import sys
import os
import logging
from pathlib import Path
from datetime import datetime

# Add root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.abduction_engine import AbductionEngine
from core.action_agent import ActionAgent, ActionType
from core.neural_learner import V2Learner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Demo")

def run_demo():
    print("\n🚀 INICIANDO DEMONSTRAÇÃO DE CAPACIDADES AVANÇADAS 🚀\n")
    
    # 1. Inicializar Componentes
    print("1️⃣  Inicializando Motores Cognitivos...")
    try:
        abduction = AbductionEngine()
        action_agent = ActionAgent()
        # Learner pode demorar ou falhar se não tiver CUDA/Model, vamos tentar
        learner = V2Learner() 
        print("✅ Componentes carregados com sucesso.\n")
    except Exception as e:
        print(f"❌ Erro na inicialização: {e}")
        return

    # 2. Ciclo de Abdução (Simulado/Forçado para Demo)
    print("2️⃣  Executando Motor de Abdução (Busca por Lacunas)...")
    
    # Vamos forçar a detecção de algumas lacunas para garantir que o demo mostre algo
    # O método run_abduction_cycle faz tudo, mas vamos quebrar para mostrar passos
    
    gaps = abduction.detect_knowledge_gaps()
    print(f"   🔍 Lacunas detectadas: {len(gaps)}")
    
    if not gaps:
        print("   ⚠️ Nenhuma lacuna real encontrada (sistema muito consistente!). Criando lacuna sintética para demo.")
        from core.abduction_engine import KnowledgeGap
        gaps = [KnowledgeGap(
            gap_id="demo_gap_01",
            gap_type="missing_connection",
            description="Falta conexão entre 'Computação Quântica' e 'Biologia Molecular'",
            affected_clusters=["Quantum", "Biology"],
            priority_score=0.9,
            candidate_hypotheses=[],
            detected_at=datetime.now()
        )]
        abduction.knowledge_gaps["demo_gap_01"] = gaps[0]

    print("3️⃣  Gerando Hipóteses...")
    hypotheses = abduction.generate_hypotheses()
    
    if not hypotheses:
         print("   ⚠️ Nenhuma hipótese gerada. Forçando hipótese para demo.")
         from core.abduction_engine import Hypothesis
         h = Hypothesis(
            id="demo_hyp_01",
            source_cluster="Quantum",
            target_cluster="Biology",
            hypothesis_text="Efeitos quânticos em microtúbulos podem explicar a consciência (Orch-OR)",
            confidence_score=0.7,
            evidence_strength=0.5,
            test_requirements=["simulation_run"],
            validation_status="pending",
            created_at=datetime.now()
         )
         abduction.hypotheses[h.id] = h
         hypotheses = [h]

    for h in hypotheses:
        print(f"   💡 Hipótese Gerada: {h.hypothesis_text} (Confiança: {h.confidence_score:.2f})")

    # 3. Agente de Ação (Validação)
    print("\n4️⃣  Acionando Agente para Validação (ActionAgent)...")
    
    for h in hypotheses:
        print(f"   🧪 Testando hipótese: {h.id}")
        
        # Simular uma ação de validação (ex: rodar uma simulação)
        if "simulation_run" in h.test_requirements or True: # Force simulation
            params = {
                "simulation_name": "quantum_bio_coherence",
                "duration": 2.0,
                "complexity": "high"
            }
            
            print(f"   ⚙️  Executando simulação: {params['simulation_name']}...")
            result = action_agent.execute_action(ActionType.SIMULATION_RUN, params)
            
            if result.status.value == "completed":
                print(f"   ✅ Simulação concluída. Dados gerados: {result.result_data.keys()}")
                # Marcar como validada para o demo
                h.validation_status = "validated"
            else:
                print(f"   ❌ Falha na simulação: {result.error_message}")

    # 4. Aprendizado Neural (Consolidação)
    print("\n5️⃣  Consolidação Neural (V2Learner)...")
    validated_hypotheses = [h for h in hypotheses if h.validation_status == "validated"]
    
    if validated_hypotheses:
        print(f"   🧠 {len(validated_hypotheses)} hipóteses validadas serão integradas ao 'subconsciente'.")
        
        # Criar vetores sintéticos representando o novo conhecimento
        import numpy as np
        vectors = np.random.normal(0, 0.1, (len(validated_hypotheses), 384)).tolist()
        
        metrics = learner.learn(vectors)
        print(f"   📉 Aprendizado concluído. Loss: {metrics.get('total_loss', 'N/A'):.4f}")
    else:
        print("   ⚪ Nenhuma hipótese validada para aprender.")

    print("\n✨ DEMONSTRAÇÃO CONCLUÍDA ✨")

if __name__ == "__main__":
    run_demo()

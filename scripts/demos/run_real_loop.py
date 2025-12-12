"""
Script de Integração: Self-Feeding Loop + AbductionEngine Real
================================================================

Conecta o loop auto-alimentado ao sistema real de raciocínio.

Uso:
    python scripts/demos/run_real_loop.py --cycles 10
"""

import sys
import os
import argparse
import logging
import random
from datetime import datetime

import numpy as np

# Path setup
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from core.loop.hypothesis_executor import HypothesisExecutor
from core.loop.feedback_collector import ActionFeedbackCollector
from core.loop.incremental_learner import IncrementalLearner
from core.loop.self_feeding_loop import SelfFeedingLoop, LoopConfig
from core.loop.loop_metrics import LoopMetrics

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Nemesis (opcional)
try:
    from core.loop.nemesis_integration import NemesisIntegration, NemesisConfig, NemesisAction
    HAS_NEMESIS = True
except ImportError:
    HAS_NEMESIS = False
    logger.warning("NemesisIntegration não disponível")


class AbductionWrapper:
    """
    Wrapper do AbductionEngine que gera gaps de fallback quando não há dados.
    Usa queries na semantic memory para criar gaps exploratórios.
    Integra com Nemesis para seleção inteligente de ações.
    """
    
    def __init__(self, abduction_engine, semantic_memory, nemesis=None):
        self.abduction_engine = abduction_engine
        self.semantic_memory = semantic_memory
        self.nemesis = nemesis  # Integração Nemesis (opcional)
        self.last_gaps = []  # Armazena gaps para compatibilidade com SelfFeedingLoop
        self.exploration_queries = [
            "neural network compression",
            "predictive coding brain",
            "free energy principle",
            "hebbian learning plasticity",
            "vector quantization",
            "active inference agent",
            "causal reasoning AI",
            "semantic memory system"
        ]
        self.current_query_idx = 0
        self.nemesis_stats = {'actions_selected': 0, 'total_efe': 0.0}
    
    def detect_knowledge_gaps(self):
        """Detecta gaps ou gera exploratórios"""
        result = []
        
        # Tentar abduction real primeiro
        if self.abduction_engine:
            try:
                gaps = self.abduction_engine.detect_knowledge_gaps()
                if gaps and len(gaps) > 0:
                    # Limitar a 1 gap por ciclo para evitar processamento excessivo
                    # Usar índice rotativo para explorar diferentes gaps
                    idx = self.current_query_idx % len(gaps)
                    self.current_query_idx += 1
                    result = [self._gap_to_dict(gaps[idx])]
                    self.last_gaps = result
                    return result
            except Exception as e:
                logger.warning(f"Abduction falhou: {e}")
        
        # Fallback: criar gap exploratório
        query = self.exploration_queries[self.current_query_idx % len(self.exploration_queries)]
        self.current_query_idx = (self.current_query_idx + 1) % len(self.exploration_queries)
        
        result = [{
            "gap_id": f"explore_{self.current_query_idx}",
            "gap_type": "exploration",
            "description": f"Explorar: {query}",
            "query": query,
            "affected_clusters": [query.split()[0]],
            "priority_score": 0.5
        }]
        self.last_gaps = result
        return result
    
    def generate_hypotheses(self, gaps=None, max_hypotheses=5):
        """Wrapper com lógica inline para garantir execução"""
        # Se gaps não passado, usar os últimos detectados
        if gaps is None:
            gaps = self.last_gaps
        try:
            # CORREÇÃO DE TIPO: Se gaps for dict, normalizar para lista
            if isinstance(gaps, dict):
                 if 'gap_id' in gaps:
                     gaps = [gaps]
                 else:
                     gaps = list(gaps.values())
                     
            with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                 f.write(f"\n--- CYCLE START ---\nCHAMADO generate_hypotheses com {len(gaps)} gaps (Tipo: {type(gaps)})\n")
            
            all_hypotheses = []
            
            for i, gap in enumerate(gaps):
                with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                    f.write(f"Iteração {i}: Processando gap {gap.get('gap_id')}\n")
                
                # LÓGICA INLINE
                hypotheses = []
                
                # 1. Tentar Engine
                if self.abduction_engine and gap.get("gap_type") != "exploration":
                    try:
                        from core.reasoning.abduction_engine import KnowledgeGap
                        kg = KnowledgeGap(
                            gap_id=gap.get("gap_id", "unknown"),
                            gap_type=gap.get("gap_type", "orphaned_cluster"),
                            description=gap.get("description", ""),
                            affected_clusters=gap.get("affected_clusters", []),
                            priority_score=gap.get("priority_score", 0.5),
                            candidate_hypotheses=[],
                            detected_at=datetime.now()
                        )
                        raw_hyps = self.abduction_engine._generate_hypotheses_for_gap(kg)
                        with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                             f.write(f"Engine retornou {len(raw_hyps)} hipoteses\n")
                        
                        if raw_hyps:
                            hypotheses = [self._hypothesis_to_dict(h) for h in raw_hyps]
                    except Exception as e:
                        with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                             f.write(f"Erro no Engine: {e}\n")
                
                # 2. Fallback se vazio
                if not hypotheses:
                    with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                        f.write("Entrando no Fallback...\n")
                    
                    query = gap.get("query", gap.get("description", ""))
                    if self.semantic_memory:
                        try:
                            results = self.semantic_memory.retrieve(query, limit=5)
                            with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                                f.write(f"Retrieve encontrou {len(results)} resultados para '{query}'\n")
                            
                            for j, r in enumerate(results[:3]):
                                content = r.get("content", "")[:100]
                                hypotheses.append({
                                    "id": f"hyp_{gap.get('gap_id')}_{j}",
                                    "hypothesis_text": f"Conexão potencial: {query} -> {content}",
                                    "source_cluster": query.split()[0],
                                    "target_cluster": "discovered",
                                    "confidence_score": 0.5,
                                    "test_requirements": []
                                })
                        except Exception as e:
                            with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                                f.write(f"Erro no Retrieve: {e}\n")

                # 3. Dummy final
                if not hypotheses:
                    with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                        f.write("Criando Dummy Hypothesis\n")
                    hypotheses.append({
                        "id": f"hyp_{gap.get('gap_id')}_dummy",
                        "hypothesis_text": f"Explorar conceito: {gap.get('description')}",
                        "source_cluster": "exploration",
                        "target_cluster": "unknown",
                        "confidence_score": 0.5,
                        "test_requirements": []
                    })

                all_hypotheses.extend(hypotheses)

            # Nemesis Selection
            if self.nemesis and all_hypotheses:
                 best = self.select_best_hypothesis(gaps[0], all_hypotheses)
                 if best:
                     return [best]

            return all_hypotheses

        except Exception as e:
            with open("DEBUG_LOG.txt", "a", encoding='utf-8') as f:
                 f.write(f"FATAL ERROR in generate_hypotheses: {e}\n")
            return []
    
    def _gap_to_dict(self, gap):
        """Converte KnowledgeGap para dict"""
        if isinstance(gap, dict):
            return gap
        return {
            "gap_id": getattr(gap, 'gap_id', 'unknown'),
            "gap_type": getattr(gap, 'gap_type', 'unknown'),
            "description": getattr(gap, 'description', ''),
            "affected_clusters": getattr(gap, 'affected_clusters', []),
            "priority_score": getattr(gap, 'priority_score', 0.5)
        }
    
    def _hypothesis_to_dict(self, hyp):
        """Converte Hypothesis para dict"""
        if isinstance(hyp, dict):
            return hyp
        return {
            "id": getattr(hyp, 'id', 'unknown'),
            "hypothesis_text": getattr(hyp, 'hypothesis_text', ''),
            "source_cluster": getattr(hyp, 'source_cluster', ''),
            "target_cluster": getattr(hyp, 'target_cluster', ''),
            "confidence_score": getattr(hyp, 'confidence_score', 0.5),
            "test_requirements": getattr(hyp, 'test_requirements', [])
        }
    
    def select_best_hypothesis(self, gap, hypotheses):
        """
        Seleciona a melhor hipótese usando Nemesis (Active Inference).
        Se Nemesis não disponível, usa confiança simples.
        """
        if not hypotheses:
            return None
        
        # Se Nemesis disponível, usar para seleção inteligente
        if self.nemesis:
            try:
                action = self.nemesis.select_action(gap, hypotheses)
                self.nemesis_stats['actions_selected'] += 1
                self.nemesis_stats['total_efe'] += action.expected_free_energy
                
                # Encontrar hipótese correspondente
                hyp_id = action.parameters.get('hypothesis_id', '')
                # Fallback se hypothesis_id não estiver nos parametros: tentar comparar id
                
                selected_hyp = None
                for h in hypotheses:
                    if h.get('id') == hyp_id:
                        selected_hyp = h
                        break
                
                # Se não achou por ID (talvez Nemesis criou uma nova ação?), pegar a primeira ou a que gerou a ação
                if not selected_hyp and hypotheses:
                     selected_hyp = hypotheses[0]
                
                if selected_hyp:
                    # Atualizar confiança com info do Nemesis
                    selected_hyp['nemesis_efe'] = action.expected_free_energy
                    selected_hyp['nemesis_selected'] = True
                    # Copiar tipo da ação do nemesis para hipoteses se necessario
                    if hasattr(action, 'type'):
                        selected_hyp['type'] = action.type
                        
                    return selected_hyp
                    
            except Exception as e:
                logger.debug(f"Nemesis selection falhou: {e}")
        
        # Fallback: maior confiança
        return max(hypotheses, key=lambda h: h.get('confidence_score', 0))


class RealLoopIntegration:
    """
    Integração real do Self-Feeding Loop com os módulos do Alexandria.
    """
    
    def __init__(self):
        self.abduction_engine = None
        self.topology_engine = None
        self.semantic_memory = None
        self.v2_learner = None
        self.loop = None
        self.nemesis = None  # Integração Nemesis
        
    def initialize(self, use_gpu: bool = False, use_nemesis: bool = False):
        """Inicializa todos os componentes reais"""
        logger.info("=" * 60)
        logger.info("🔄 INICIALIZANDO SELF-FEEDING LOOP REAL")
        logger.info("=" * 60)
        
        # 1. Topology Engine
        logger.info("📐 Carregando Topology Engine...")
        try:
            from core.topology.topology_engine import TopologyEngine
            self.topology_engine = TopologyEngine()
            self.topology_engine = TopologyEngine()
            logger.info("   Topology Engine OK")
        except Exception as e:
            logger.warning(f"   Topology Engine falhou: {e}")
        
        # 2. Semantic Memory
        logger.info("🧠 Carregando Semantic Memory...")
        try:
            from core.memory.semantic_memory import SemanticFileSystem
            self.semantic_memory = SemanticFileSystem(self.topology_engine)
            self.semantic_memory = SemanticFileSystem(self.topology_engine)
            logger.info("   Semantic Memory OK")
        except Exception as e:
            logger.warning(f"   Semantic Memory falhou: {e}")
        
        # 3. Abduction Engine
        logger.info("🔮 Carregando Abduction Engine...")
        try:
            from core.reasoning.abduction_engine import AbductionEngine
            self.abduction_engine = AbductionEngine()
            self.abduction_engine = AbductionEngine()
            logger.info("   Abduction Engine OK")
        except Exception as e:
            logger.warning(f"   Abduction Engine falhou: {e}")
        
        # 4. V2 Learner
        logger.info("🎓 Carregando V2 Learner...")
        try:
            from core.reasoning.neural_learner import V2Learner
            device = "cuda" if use_gpu else "cpu"
            self.v2_learner = V2Learner(device=device)
            self.v2_learner = V2Learner(device=device)
            logger.info(f"   V2 Learner OK (device={device})")
        except Exception as e:
            logger.warning(f"   V2 Learner falhou: {e}")
        
        # 5. [OPCIONAL] Nemesis Integration
        if use_nemesis and HAS_NEMESIS:
            logger.info("🧠 Carregando Nemesis Integration...")
            try:
                config = NemesisConfig(
                    observation_dim=384,
                    use_active_inference=True,
                    use_predictive_coding=True,
                    use_free_energy=True,
                    use_meta_hebbian=True
                )
                self.nemesis = NemesisIntegration(config, self.topology_engine)
                self.nemesis = NemesisIntegration(config, self.topology_engine)
                logger.info("   Nemesis Integration OK")
            except Exception as e:
                logger.warning(f"   Nemesis falhou: {e}")
                self.nemesis = None
        
        self.setup_loop()
        return True

    def _on_action_complete(self, hypothesis, result, feedback):
        """Callback para atualizar Nemesis após ação"""
        if not self.nemesis:
            return
            
        try:
            # 1. Recuperar/Gerar embedding da observação
            observation = np.zeros(384) 
            if result.evidence_found and self.topology_engine:
                 # Assumindo que evidence_found é uma lista de textos
                 # Pegar o primeiro para simplificar
                 item = result.evidence_found[0]
                 text = item if isinstance(item, str) else str(item)
                 # Usar encode que retorna lista, pegar primeiro vetor
                 observation = self.topology_engine.encode([text])[0]
            
            # 2. Reward
            reward = feedback.reward_signal
            
            # 3. Reconstruir Ação
            # Se Nemesis selecionou, 'nemesis_efe' deve estar na hipótese
            action_type = str(hypothesis.get('type', 'VALIDATE_CAUSAL'))
            
            action = NemesisAction(
                action_type=action_type,
                target=hypothesis.get('target_cluster', 'unknown'),
                parameters=hypothesis,
                expected_free_energy=hypothesis.get('nemesis_efe', 0.0)
            )
            
            # 4. Atualizar Nemesis
            self.nemesis.update_after_action(action, observation, reward)
            
            # Debug info
            efe = hypothesis.get('nemesis_efe', 0.0)
            logger.debug(f"Nemesis Updated | Reward: {reward:.3f} | EFE: {efe:.3f}")
            
        except Exception as e:
            logger.warning(f"Erro no callback do Nemesis: {e}")

    def setup_loop(self):
        """Configura o loop com os componentes carregados"""
        if not self.abduction_engine:
            # Se falhou abduction, criar dummy wrapper se possível ou falhar
            pass
            
        # 1. Configurar componentes do loop
        executor = HypothesisExecutor(
            semantic_memory=self.semantic_memory,
            topology_engine=self.topology_engine,
            high_confidence_threshold=0.6  # Criar conexões com confiança >= 0.6
        )
        
        collector = ActionFeedbackCollector(
            topology_engine=self.topology_engine
        )
        
        learner = IncrementalLearner(
            v2_learner=self.v2_learner,
            batch_threshold=5,
            auto_save=True
        )
        
        config = LoopConfig(
            max_cycles=100,
            max_hypotheses_per_cycle=5,
            stop_on_convergence=True,
            convergence_threshold=0.01,
            min_confidence_threshold=0.2,
            metrics_save_path="data/real_loop_metrics.json"
        )
        
        # Criar wrapper do abduction
        abduction_wrapper = AbductionWrapper(
            self.abduction_engine, 
            self.semantic_memory,
            nemesis=self.nemesis
        )
        
        self.loop = SelfFeedingLoop(
            abduction_engine=abduction_wrapper,
            hypothesis_executor=executor,
            feedback_collector=collector,
            incremental_learner=learner,
            config=config,
            on_action_complete=self._on_action_complete
        )
        
        logger.info("   Self-Feeding Loop montado")
        
    def run(self, max_cycles: int = 10) -> dict:
        """Executa o loop"""
        if not self.loop:
            logger.error("Loop não inicializado. Chame initialize() primeiro.")
            return {}
        
        logger.info(f"\nExecutando {max_cycles} ciclos...\n")
        
        try:
            results = self.loop.run_continuous(max_cycles=max_cycles)
            self._print_results(results)
            return results
        except Exception as e:
            logger.error(f"Erro na execução do loop: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _print_results(self, results: dict):
        """Imprime resultados formatados"""
        print("\n" + "=" * 60)
        print("📊 RESULTADOS DO LOOP REAL")
        print("=" * 60)
        
        print(f"\nCiclos executados: {results.get('cycles_run', 0)}")
        print(f"Tempo total: {results.get('total_time_seconds', 0):.2f}s")
        print(f"Convergiu: {results.get('converged', False)}")
        
        summary = results.get('metrics_summary', {})
        print(f"\nMetricas:")
        print(f"   Gaps detectados: {summary.get('total_gaps', 0)}")
        print(f"   Hipóteses geradas: {summary.get('total_hypotheses', 0)}")
        print(f"   Ações executadas: {summary.get('total_actions', 0)}")
        print(f"   Taxa de sucesso: {summary.get('success_rate', 0):.1%}")
        print(f"   Evidências: {summary.get('total_evidence', 0)}")
        print(f"   Conexões: {summary.get('total_connections', 0)}")
        print(f"   Eventos de aprendizado: {summary.get('total_learning_events', 0)}")
        
        learner_stats = results.get('learner_stats', {})
        print(f"\nAprendizado:")
        print(f"   Embeddings aprendidos: {learner_stats.get('total_learned', 0)}")
        print(f"   Última loss: {learner_stats.get('last_loss', 0):.4f}")
        
        if self.nemesis:
            print(f"\nNemesis Stats:")
            print(f"   Free Energy (current): {self.nemesis.state.free_energy:.4f}")
            print(f"   Prediction Error: {self.nemesis.state.prediction_error:.4f}")
        
        print(f"\nMetricas salvas em: data/real_loop_metrics.json")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Self-Feeding Loop Real')
    parser.add_argument('--cycles', type=int, default=10, help='Número de ciclos')
    parser.add_argument('--gpu', action='store_true', help='Usar GPU')
    parser.add_argument('--use-nemesis', action='store_true', help='Usar integração Nemesis (Active Inference)')
    args = parser.parse_args()
    
    integration = RealLoopIntegration()
    
    if integration.initialize(use_gpu=args.gpu, use_nemesis=args.use_nemesis):
        integration.run(max_cycles=args.cycles)
    else:
        logger.error("Falha na inicialização")


if __name__ == "__main__":
    main()

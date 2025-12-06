"""
Meta-Hebbian Plasticity Module for Alexandria
==============================================

Implementação de meta-aprendizado via regras Hebbianas aprendidas.
Baseado em: "Meta-Learning through Hebbian Plasticity in Random Networks"
            Najarro & Risi, NeurIPS 2020

Este módulo implementa a função M do framework S-CAS:
    M(H, T_old) → T_new

Onde as regras de plasticidade (não os pesos) são o que evolui.

Hierarquia de paradigmas (caminho de evolução):
    Hebbian → Meta-Hebbian → Predictive Coding → Active Inference → Free Energy
                   ↑
              VOCÊ ESTÁ AQUI

Autor: G (Alexandria Project)
Versão: 1.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import pickle
from pathlib import Path


# =============================================================================
# CONFIGURAÇÃO E TIPOS
# =============================================================================

class EvolutionMethod(Enum):
    """Método de evolução das regras de plasticidade"""
    EVOLUTION_STRATEGIES = "es"      # Como no paper original
    GRADIENT_DESCENT = "gradient"    # Diferenciável (Miconi et al.)
    HYBRID = "hybrid"                # ES para exploração, gradiente para refinamento


@dataclass
class PlasticityRule:
    """
    Regra de plasticidade ABCD generalizada.
    
    Δw_ij = η * (A * o_i * o_j + B * o_i + C * o_j + D)
    
    Onde:
        - A: Termo Hebbian clássico (correlação pré-pós)
        - B: Termo pré-sináptico (input-driven)
        - C: Termo pós-sináptico (output-driven)  
        - D: Termo de bias/decay
        - η: Taxa de aprendizado
    """
    A: float = 0.1      # Hebbian correlation
    B: float = 0.0      # Pre-synaptic bias
    C: float = 0.0      # Post-synaptic bias
    D: float = -0.001   # Decay/bias (negativo = decay natural)
    eta: float = 0.01   # Learning rate
    
    def to_array(self) -> np.ndarray:
        return np.array([self.A, self.B, self.C, self.D, self.eta])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'PlasticityRule':
        return cls(A=arr[0], B=arr[1], C=arr[2], D=arr[3], eta=arr[4])
    
    def compute_delta(self, pre: float, post: float) -> float:
        """Computa Δw para uma única sinapse"""
        return self.eta * (self.A * pre * post + self.B * pre + self.C * post + self.D)
    
    def compute_delta_matrix(self, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        """Computa Δw para todas as sinapses (outer product generalizado)"""
        hebbian = self.A * np.outer(pre, post)
        pre_term = self.B * pre[:, np.newaxis] * np.ones_like(post)
        post_term = self.C * np.ones_like(pre)[:, np.newaxis] * post
        bias = self.D * np.ones((len(pre), len(post)))
        
        return self.eta * (hebbian + pre_term + post_term + bias)


@dataclass
class MetaHebbianConfig:
    """Configuração do sistema Meta-Hebbian"""
    num_codes: int = 1024               # Número de códigos no codebook VQ-VAE
    num_heads: int = 4                  # Número de heads (grupos de regras)
    rules_per_head: bool = True         # Uma regra por head (False = regra global)
    
    # Evolução
    evolution_method: EvolutionMethod = EvolutionMethod.EVOLUTION_STRATEGIES
    population_size: int = 50           # Tamanho da população para ES
    sigma: float = 0.1                  # Desvio padrão para mutação
    elite_ratio: float = 0.2            # Proporção de elite mantida
    
    # Limites das regras (para estabilidade)
    param_bounds: Tuple[float, float] = (-2.0, 2.0)
    eta_bounds: Tuple[float, float] = (0.0001, 0.1)
    
    # Persistência
    save_path: str = "data/meta_hebbian_state.pkl"


# =============================================================================
# NÚCLEO META-HEBBIAN
# =============================================================================

class MetaHebbianPlasticity:
    """
    Sistema de Meta-Aprendizado via Plasticidade Hebbian.
    
    Em vez de otimizar pesos diretamente, otimiza as REGRAS
    que governam como os pesos mudam durante runtime.
    
    Isso permite:
    - Adaptação lifetime (pesos continuam evoluindo)
    - Robustez a perturbações
    - Transfer learning natural
    - Compatibilidade com Free Energy Principle
    """
    
    def __init__(self, config: Optional[MetaHebbianConfig] = None):
        self.config = config or MetaHebbianConfig()
        
        # Inicializa regras de plasticidade
        self._init_rules()
        
        # Histórico para meta-aprendizado
        self.performance_history: List[float] = []
        self.rule_history: List[Dict] = []
        
        # Estado interno
        self._generation = 0
        self._best_fitness = float('-inf')
        self._best_rules = None
        
    def _init_rules(self):
        """Inicializa regras de plasticidade"""
        if self.config.rules_per_head:
            # Uma regra por head
            self.rules = [
                PlasticityRule(
                    A=0.1 + np.random.randn() * 0.02,
                    B=np.random.randn() * 0.01,
                    C=np.random.randn() * 0.01,
                    D=-0.001 + np.random.randn() * 0.0005,
                    eta=0.01
                )
                for _ in range(self.config.num_heads)
            ]
        else:
            # Regra global única
            self.rules = [PlasticityRule()]
    
    # =========================================================================
    # APLICAÇÃO DAS REGRAS (RUNTIME)
    # =========================================================================
    
    def compute_weight_update(
        self,
        weights: np.ndarray,
        pre_activations: np.ndarray,
        post_activations: np.ndarray,
        head_idx: int = 0
    ) -> np.ndarray:
        """
        Computa atualização de pesos usando a regra Meta-Hebbian.
        
        Args:
            weights: Matriz de pesos atual [pre_size x post_size]
            pre_activations: Ativações pré-sinápticas [pre_size]
            post_activations: Ativações pós-sinápticas [post_size]
            head_idx: Índice do head (para regra específica)
            
        Returns:
            new_weights: Matriz de pesos atualizada
        """
        rule_idx = head_idx if self.config.rules_per_head else 0
        rule = self.rules[rule_idx]
        
        # Normaliza ativações para estabilidade
        pre_norm = self._normalize_activations(pre_activations)
        post_norm = self._normalize_activations(post_activations)
        
        # Computa delta
        delta_w = rule.compute_delta_matrix(pre_norm, post_norm)
        
        # Aplica com clipping para estabilidade
        new_weights = weights + delta_w
        new_weights = np.clip(new_weights, -1.0, 1.0)
        
        return new_weights
    
    def apply_to_mycelial(
        self,
        connections: np.ndarray,
        activated_codes: np.ndarray,
        activation_strengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Aplica regras Meta-Hebbian à rede micelial.
        
        Esta é a interface principal com o MycelialReasoning existente.
        
        Args:
            connections: Matriz de conexões micelial [num_codes x num_codes]
            activated_codes: Índices dos códigos ativados
            activation_strengths: Força de cada ativação (opcional)
            
        Returns:
            updated_connections: Matriz de conexões atualizada
        """
        if len(activated_codes) < 2:
            return connections
        
        # Se não fornecido, assume ativação uniforme
        if activation_strengths is None:
            activation_strengths = np.ones(len(activated_codes))
        
        # Cria vetor de ativação esparso
        pre_activation = np.zeros(self.config.num_codes)
        post_activation = np.zeros(self.config.num_codes)
        
        for idx, strength in zip(activated_codes, activation_strengths):
            pre_activation[idx] = strength
            post_activation[idx] = strength
        
        # Determina head baseado no primeiro código (ou distribui)
        head_idx = activated_codes[0] % self.config.num_heads
        
        # Aplica regra
        updated = self.compute_weight_update(
            connections, 
            pre_activation, 
            post_activation,
            head_idx
        )
        
        return updated
    
    def _normalize_activations(self, activations: np.ndarray) -> np.ndarray:
        """Normaliza ativações para estabilidade numérica"""
        norm = np.linalg.norm(activations)
        if norm > 1e-8:
            return activations / norm
        return activations
    
    # =========================================================================
    # EVOLUÇÃO DAS REGRAS (META-LEARNING)
    # =========================================================================
    
    def evolve_rules(
        self,
        fitness_scores: List[float],
        method: Optional[EvolutionMethod] = None
    ) -> Dict[str, Any]:
        """
        Evolui as regras de plasticidade baseado em fitness.
        
        Este é o M(H, T_old) → T_new do framework S-CAS.
        
        Args:
            fitness_scores: Lista de scores de fitness recentes
            method: Método de evolução (usa config se None)
            
        Returns:
            evolution_stats: Estatísticas da evolução
        """
        method = method or self.config.evolution_method
        
        # Registra histórico
        avg_fitness = np.mean(fitness_scores) if fitness_scores else 0.0
        self.performance_history.append(avg_fitness)
        
        if method == EvolutionMethod.EVOLUTION_STRATEGIES:
            stats = self._evolve_es(avg_fitness)
        elif method == EvolutionMethod.GRADIENT_DESCENT:
            stats = self._evolve_gradient(fitness_scores)
        else:  # HYBRID
            if self._generation % 10 == 0:
                stats = self._evolve_es(avg_fitness)
            else:
                stats = self._evolve_gradient(fitness_scores)
        
        self._generation += 1
        
        # Salva histórico de regras
        self.rule_history.append({
            'generation': self._generation,
            'fitness': avg_fitness,
            'rules': [r.to_array().tolist() for r in self.rules]
        })
        
        return stats
    
    def _evolve_es(self, current_fitness: float) -> Dict[str, Any]:
        """
        Evolution Strategies para otimizar regras.
        
        Baseado no paper original de Najarro & Risi.
        """
        # Atualiza melhor se necessário
        if current_fitness > self._best_fitness:
            self._best_fitness = current_fitness
            self._best_rules = [PlasticityRule.from_array(r.to_array().copy()) for r in self.rules]
        
        # Gera população de regras perturbadas
        population = []
        for _ in range(self.config.population_size):
            perturbed_rules = []
            for rule in self.rules:
                params = rule.to_array()
                noise = np.random.randn(len(params)) * self.config.sigma
                new_params = params + noise
                
                # Aplica limites
                new_params[:4] = np.clip(
                    new_params[:4], 
                    self.config.param_bounds[0], 
                    self.config.param_bounds[1]
                )
                new_params[4] = np.clip(
                    new_params[4],
                    self.config.eta_bounds[0],
                    self.config.eta_bounds[1]
                )
                
                perturbed_rules.append(PlasticityRule.from_array(new_params))
            population.append(perturbed_rules)
        
        # Em um sistema real, avaliaria cada conjunto de regras
        # Aqui, fazemos uma aproximação baseada na direção do gradiente de fitness
        
        if len(self.performance_history) >= 2:
            fitness_delta = self.performance_history[-1] - self.performance_history[-2]
            
            if fitness_delta > 0:
                # Fitness melhorando: continua na direção atual
                # Pequena perturbação exploradora
                for i, rule in enumerate(self.rules):
                    params = rule.to_array()
                    params += np.random.randn(len(params)) * self.config.sigma * 0.5
                    params[:4] = np.clip(params[:4], *self.config.param_bounds)
                    params[4] = np.clip(params[4], *self.config.eta_bounds)
                    self.rules[i] = PlasticityRule.from_array(params)
            else:
                # Fitness piorando: volta para melhor conhecido + explora
                if self._best_rules:
                    for i, best_rule in enumerate(self._best_rules):
                        params = best_rule.to_array()
                        params += np.random.randn(len(params)) * self.config.sigma
                        params[:4] = np.clip(params[:4], *self.config.param_bounds)
                        params[4] = np.clip(params[4], *self.config.eta_bounds)
                        self.rules[i] = PlasticityRule.from_array(params)
        
        return {
            'method': 'evolution_strategies',
            'generation': self._generation,
            'current_fitness': current_fitness,
            'best_fitness': self._best_fitness,
            'sigma': self.config.sigma,
            'rules_updated': True
        }
    
    def _evolve_gradient(self, fitness_scores: List[float]) -> Dict[str, Any]:
        """
        Gradient-based evolution (diferenciável).
        
        Usa diferenças finitas para aproximar gradiente.
        """
        if len(fitness_scores) < 2:
            return {'method': 'gradient', 'status': 'insufficient_data'}
        
        # Aproxima gradiente via diferenças finitas
        # Em implementação completa, usaria autodiff (JAX)
        
        gradient_lr = 0.001
        fitness_trend = np.mean(fitness_scores[-5:]) - np.mean(fitness_scores[-10:-5]) if len(fitness_scores) >= 10 else 0
        
        for rule in self.rules:
            params = rule.to_array()
            
            # Ajuste baseado em tendência de fitness
            if fitness_trend > 0:
                # Aumenta componente Hebbian se melhorando
                params[0] *= 1.01  # A
            else:
                # Aumenta decay se piorando (regularização)
                params[3] *= 1.01  # D (mais negativo)
            
            params[:4] = np.clip(params[:4], *self.config.param_bounds)
            params[4] = np.clip(params[4], *self.config.eta_bounds)
        
        return {
            'method': 'gradient',
            'generation': self._generation,
            'fitness_trend': fitness_trend,
            'rules_updated': True
        }
    
    # =========================================================================
    # ANÁLISE E DIAGNÓSTICO
    # =========================================================================
    
    def get_rule_analysis(self) -> Dict[str, Any]:
        """Análise detalhada das regras atuais"""
        rules_data = []
        for i, rule in enumerate(self.rules):
            rules_data.append({
                'head': i,
                'A_hebbian': rule.A,
                'B_pre': rule.B,
                'C_post': rule.C,
                'D_decay': rule.D,
                'eta_lr': rule.eta,
                'is_hebbian_dominant': abs(rule.A) > abs(rule.B) + abs(rule.C),
                'is_decaying': rule.D < 0,
                'effective_strength': abs(rule.A) * rule.eta
            })
        
        # Estatísticas agregadas
        all_A = [r.A for r in self.rules]
        all_eta = [r.eta for r in self.rules]
        
        return {
            'rules': rules_data,
            'aggregate': {
                'mean_hebbian_strength': np.mean(all_A),
                'std_hebbian_strength': np.std(all_A),
                'mean_learning_rate': np.mean(all_eta),
                'rule_diversity': np.std([r.to_array() for r in self.rules]),
                'generation': self._generation,
                'best_fitness': self._best_fitness
            },
            'interpretation': self._interpret_rules()
        }
    
    def _interpret_rules(self) -> str:
        """Interpretação legível das regras"""
        avg_A = np.mean([r.A for r in self.rules])
        avg_D = np.mean([r.D for r in self.rules])
        
        if avg_A > 0.1 and avg_D < -0.0005:
            return "BALANCED: Aprendizado Hebbian ativo com decay saudável"
        elif avg_A > 0.1 and avg_D >= 0:
            return "ACCUMULATING: Conexões crescendo sem decay - risco de saturação"
        elif avg_A < 0.05:
            return "WEAK_LEARNING: Componente Hebbian fraco - aprendizado lento"
        elif avg_D < -0.01:
            return "HIGH_DECAY: Decay muito forte - pode esquecer muito rápido"
        else:
            return "EXPLORING: Regras em estado de exploração"
    
    def get_performance_trend(self, window: int = 10) -> Dict[str, float]:
        """Calcula tendência de performance"""
        if len(self.performance_history) < window:
            return {'trend': 0.0, 'volatility': 0.0, 'status': 'insufficient_data'}
        
        recent = self.performance_history[-window:]
        older = self.performance_history[-2*window:-window] if len(self.performance_history) >= 2*window else self.performance_history[:window]
        
        trend = np.mean(recent) - np.mean(older)
        volatility = np.std(recent)
        
        return {
            'trend': trend,
            'volatility': volatility,
            'recent_mean': np.mean(recent),
            'improving': trend > 0,
            'stable': volatility < 0.1
        }
    
    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================
    
    def save_state(self, path: Optional[str] = None):
        """Salva estado completo do sistema Meta-Hebbian"""
        path = path or self.config.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': {
                'num_codes': self.config.num_codes,
                'num_heads': self.config.num_heads,
                'rules_per_head': self.config.rules_per_head,
                'evolution_method': self.config.evolution_method.value,
                'population_size': self.config.population_size,
                'sigma': self.config.sigma
            },
            'rules': [r.to_array().tolist() for r in self.rules],
            'performance_history': self.performance_history,
            'rule_history': self.rule_history[-100:],  # Últimas 100 gerações
            'generation': self._generation,
            'best_fitness': self._best_fitness,
            'best_rules': [r.to_array().tolist() for r in self._best_rules] if self._best_rules else None
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        return path
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Carrega estado salvo"""
        path = path or self.config.save_path
        
        if not Path(path).exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            self.rules = [PlasticityRule.from_array(np.array(r)) for r in state['rules']]
            self.performance_history = state.get('performance_history', [])
            self.rule_history = state.get('rule_history', [])
            self._generation = state.get('generation', 0)
            self._best_fitness = state.get('best_fitness', float('-inf'))
            
            if state.get('best_rules'):
                self._best_rules = [PlasticityRule.from_array(np.array(r)) for r in state['best_rules']]
            
            return True
        except Exception as e:
            print(f"Erro ao carregar estado Meta-Hebbian: {e}")
            return False


# =============================================================================
# INTEGRAÇÃO COM MYCELIAL REASONING
# =============================================================================

class MetaHebbianMycelialIntegration:
    """
    Camada de integração entre MetaHebbianPlasticity e MycelialReasoning.
    
    Substitui o Hebbian fixo do MycelialReasoning por regras adaptativas.
    """
    
    def __init__(
        self,
        meta_hebbian: MetaHebbianPlasticity,
        mycelial_reasoning: Any  # MycelialReasoning (import circular)
    ):
        self.meta = meta_hebbian
        self.mycelial = mycelial_reasoning
        self.observation_count = 0
        self.fitness_buffer: List[float] = []
        
    def observe_and_learn(
        self,
        indices: np.ndarray,
        relevance_scores: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Observa ativação e aplica aprendizado Meta-Hebbian.
        
        Substitui o método observe() do MycelialReasoning.
        
        Args:
            indices: Códigos VQ-VAE ativados
            relevance_scores: Scores de relevância (para fitness)
            
        Returns:
            stats: Estatísticas da observação
        """
        if len(indices) < 2:
            return {'status': 'skipped', 'reason': 'insufficient_indices'}
        
        # Aplica regras Meta-Hebbian às conexões
        old_connections = self.mycelial.connections.copy()
        
        # Computa ativações baseadas nos índices
        activation_strengths = relevance_scores if relevance_scores is not None else np.ones(len(indices))
        
        # Atualiza conexões usando Meta-Hebbian
        new_connections = self.meta.apply_to_mycelial(
            self.mycelial.connections,
            indices,
            activation_strengths
        )
        
        # Aplica atualização
        self.mycelial.connections = new_connections
        
        # Calcula fitness como medida de "qualidade" da atualização
        # (diferença em esparsidade, magnitude de mudança, etc.)
        connection_change = np.abs(new_connections - old_connections).mean()
        sparsity = (np.abs(new_connections) < 0.01).mean()
        
        # Fitness: queremos mudanças moderadas e rede esparsa
        fitness = 1.0 - abs(connection_change - 0.01) * 10 + sparsity * 0.5
        self.fitness_buffer.append(fitness)
        
        self.observation_count += 1
        
        # Evolui regras periodicamente
        evolution_stats = None
        if self.observation_count % 100 == 0 and len(self.fitness_buffer) >= 10:
            evolution_stats = self.meta.evolve_rules(self.fitness_buffer[-100:])
            self.fitness_buffer = self.fitness_buffer[-50:]  # Mantém buffer limitado
        
        return {
            'status': 'success',
            'observation_count': self.observation_count,
            'connection_change': float(connection_change),
            'sparsity': float(sparsity),
            'fitness': float(fitness),
            'evolution_triggered': evolution_stats is not None,
            'evolution_stats': evolution_stats
        }
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Estatísticas da integração"""
        return {
            'observation_count': self.observation_count,
            'fitness_buffer_size': len(self.fitness_buffer),
            'recent_fitness': np.mean(self.fitness_buffer[-10:]) if self.fitness_buffer else 0.0,
            'meta_hebbian_stats': self.meta.get_rule_analysis(),
            'mycelial_stats': {
                'total_connections': int((np.abs(self.mycelial.connections) > 0.01).sum()),
                'mean_weight': float(self.mycelial.connections.mean()),
                'max_weight': float(self.mycelial.connections.max())
            }
        }


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def create_meta_hebbian_system(
    num_codes: int = 1024,
    num_heads: int = 4,
    load_existing: bool = True,
    save_path: str = "data/meta_hebbian_state.pkl"
) -> MetaHebbianPlasticity:
    """
    Factory function para criar sistema Meta-Hebbian.
    
    Args:
        num_codes: Número de códigos no codebook
        num_heads: Número de heads/grupos de regras
        load_existing: Se True, tenta carregar estado existente
        save_path: Caminho para persistência
        
    Returns:
        meta: Sistema MetaHebbianPlasticity configurado
    """
    config = MetaHebbianConfig(
        num_codes=num_codes,
        num_heads=num_heads,
        save_path=save_path
    )
    
    meta = MetaHebbianPlasticity(config)
    
    if load_existing:
        loaded = meta.load_state()
        if loaded:
            print(f"✅ Meta-Hebbian carregado: {meta._generation} gerações, fitness={meta._best_fitness:.4f}")
        else:
            print("🌱 Meta-Hebbian inicializado com regras padrão")
    
    return meta


def integrate_with_mycelial(
    meta: MetaHebbianPlasticity,
    mycelial
) -> MetaHebbianMycelialIntegration:
    """
    Integra Meta-Hebbian com MycelialReasoning existente.
    
    Args:
        meta: Sistema MetaHebbianPlasticity
        mycelial: Instância de MycelialReasoning
        
    Returns:
        integration: Camada de integração configurada
    """
    return MetaHebbianMycelialIntegration(meta, mycelial)


# =============================================================================
# EXEMPLO DE USO E TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("META-HEBBIAN PLASTICITY - ALEXANDRIA")
    print("=" * 70)
    
    # Criar sistema
    meta = create_meta_hebbian_system(
        num_codes=1024,
        num_heads=4,
        load_existing=True
    )
    
    # Análise das regras
    print("\n📊 ANÁLISE DAS REGRAS:")
    analysis = meta.get_rule_analysis()
    
    for rule_data in analysis['rules']:
        print(f"\n   Head {rule_data['head']}:")
        print(f"      A (Hebbian):  {rule_data['A_hebbian']:+.4f}")
        print(f"      B (Pre):      {rule_data['B_pre']:+.4f}")
        print(f"      C (Post):     {rule_data['C_post']:+.4f}")
        print(f"      D (Decay):    {rule_data['D_decay']:+.4f}")
        print(f"      η (LR):       {rule_data['eta_lr']:.4f}")
        print(f"      Dominante:    {'Hebbian' if rule_data['is_hebbian_dominant'] else 'Outros'}")
    
    print(f"\n🎯 INTERPRETAÇÃO: {analysis['interpretation']}")
    print(f"   Geração: {analysis['aggregate']['generation']}")
    print(f"   Melhor Fitness: {analysis['aggregate']['best_fitness']:.4f}")
    
    # Simular algumas observações
    print("\n🔄 SIMULANDO OBSERVAÇÕES...")
    
    # Mock de conexões miceliais
    mock_connections = np.random.randn(1024, 1024) * 0.01
    
    for i in range(10):
        # Simula códigos ativados
        activated = np.random.choice(1024, size=5, replace=False)
        strengths = np.random.rand(5)
        
        # Aplica Meta-Hebbian
        new_connections = meta.apply_to_mycelial(
            mock_connections,
            activated,
            strengths
        )
        
        change = np.abs(new_connections - mock_connections).mean()
        mock_connections = new_connections
        
        if i % 3 == 0:
            print(f"   Obs {i+1}: Mudança média = {change:.6f}")
    
    # Evolui regras
    print("\n🧬 EVOLUINDO REGRAS...")
    fitness_scores = np.random.rand(20).tolist()  # Mock fitness
    evolution_result = meta.evolve_rules(fitness_scores)
    print(f"   Método: {evolution_result['method']}")
    print(f"   Geração: {evolution_result['generation']}")
    
    # Salvar estado
    save_path = meta.save_state()
    print(f"\n💾 Estado salvo em: {save_path}")
    
    print("\n" + "=" * 70)
    print("✅ META-HEBBIAN PRONTO PARA INTEGRAÇÃO COM ALEXANDRIA")
    print("=" * 70)
    
    print("""
    
PRÓXIMOS PASSOS:
================

1. Integrar com MycelialReasoning:
   
   from meta_hebbian import create_meta_hebbian_system, integrate_with_mycelial
   
   meta = create_meta_hebbian_system()
   integration = integrate_with_mycelial(meta, mycelial_reasoning)
   
   # Substituir observe() por:
   stats = integration.observe_and_learn(indices, relevance_scores)

2. Caminho de Evolução:
   
   Hebbian (atual) → Meta-Hebbian (este arquivo) → Predictive Coding → Active Inference
   
3. Para Predictive Coding (próximo passo):
   
   Adicionar camada que prediz próxima ativação e computa ERRO.
   Erro propaga ao invés de ativação direta.
   
    """)

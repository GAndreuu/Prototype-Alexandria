"""
PreStructuralField: Wrapper que unifica o Campo com VQ-VAE e Mycelial
======================================================================

Este módulo é a "cola" que conecta:
- DynamicManifold ← VQ-VAE (códigos definem pontos)
- CycleDynamics → Mycelial (cristalização alimenta grafo)
- FreeEnergyField ↔ VariationalFreeEnergy (sync)

Autor: Alexandria Project
Data: 2025-12-08
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from .manifold import DynamicManifold, ManifoldConfig, ManifoldPoint
from .metric import RiemannianMetric, MetricConfig
from .free_energy_field import FreeEnergyField, FieldConfig, FieldState
from .geodesic_flow import GeodesicFlow, GeodesicConfig
from .cycle_dynamics import CycleDynamics, CycleConfig, CycleState

logger = logging.getLogger(__name__)


@dataclass
class PreStructuralConfig:
    """Configuração unificada do Campo Pré-Estrutural."""
    # Manifold
    base_dim: int = 384
    max_expansion: int = 64
    
    # Metric
    deformation_radius: float = 0.3
    deformation_strength: float = 0.5
    
    # Field
    temperature: float = 1.0
    
    # Cycle
    configuration_steps: int = 20
    expansion_threshold: float = 0.7
    compression_threshold: float = 0.3
    
    # Geodesic
    max_geodesic_steps: int = 20
    use_scipy: bool = False  # False = mais rápido


class PreStructuralField:
    """
    Campo Pré-Estrutural completo.
    
    Unifica todos os componentes e conecta com VQ-VAE e Mycelial.
    
    Uso:
        field = PreStructuralField()
        field.connect_vqvae(vqvae_model)
        field.connect_mycelial(mycelial_reasoning)
        
        # Trigger um conceito
        state = field.trigger(embedding)
        
        # Propagar e ver atratores
        states = field.propagate(state, steps=5)
        
        # Cristalizar em grafo
        graph = field.crystallize()
    """
    
    def __init__(self, config: Optional[PreStructuralConfig] = None):
        self.config = config or PreStructuralConfig()
        
        # Inicializa componentes
        self._init_components()
        
        # Conexões externas (opcionais)
        self.vqvae = None
        self.mycelial = None
        self.variational_fe = None
        
        # Histórico
        self.trigger_count = 0
        self.cycle_history: List[CycleState] = []
        
        logger.info(f"PreStructuralField inicializado: {self.config.base_dim}D")
    
    def _init_components(self):
        """Inicializa todos os componentes do campo."""
        
        # 1. Manifold
        manifold_config = ManifoldConfig(
            base_dim=self.config.base_dim,
            max_expansion=self.config.max_expansion
        )
        self.manifold = DynamicManifold(manifold_config)
        
        # 2. Metric
        metric_config = MetricConfig(
            deformation_radius=self.config.deformation_radius,
            deformation_strength=self.config.deformation_strength
        )
        self.metric = RiemannianMetric(self.manifold, metric_config)
        
        # 3. Field
        field_config = FieldConfig(
            temperature=self.config.temperature
        )
        self.field = FreeEnergyField(self.manifold, self.metric, field_config)
        
        # 4. Geodesic Flow
        flow_config = GeodesicConfig(
            max_steps=self.config.max_geodesic_steps,
            use_scipy_integrator=self.config.use_scipy
        )
        self.flow = GeodesicFlow(self.manifold, self.metric, flow_config)
        
        # 5. Cycle Dynamics
        cycle_config = CycleConfig(
            configuration_steps=self.config.configuration_steps,
            expansion_threshold=self.config.expansion_threshold,
            compression_threshold=self.config.compression_threshold
        )
        self.cycle = CycleDynamics(
            self.manifold, self.metric, self.field, self.flow, cycle_config
        )
    
    # =========================================================================
    # CONEXÕES EXTERNAS
    # =========================================================================
    
    def connect_vqvae(self, vqvae_model):
        """
        Conecta com modelo VQ-VAE.
        
        Args:
            vqvae_model: Instância de MonolithV13 ou MonolithWiki
        """
        self.vqvae = vqvae_model
        
        # Extrai codebook como anchor points
        if hasattr(vqvae_model, 'get_codebook'):
            codebook = vqvae_model.get_codebook()
            self.manifold.set_anchor_points(codebook)
            logger.info(f"VQ-VAE conectado: codebook shape {codebook.shape}")
        else:
            logger.warning("VQ-VAE não tem get_codebook(), anchor points não setados")
    
    def connect_mycelial(self, mycelial_reasoning):
        """
        Conecta com MycelialReasoning.
        
        Args:
            mycelial_reasoning: Instância de MycelialReasoning
        """
        self.mycelial = mycelial_reasoning
        logger.info("Mycelial conectado")
    
    def connect_variational_fe(self, vfe):
        """
        Conecta com VariationalFreeEnergy existente.
        
        Permite sincronizar beliefs e componentes de F.
        """
        self.variational_fe = vfe
        self.field.vfe = vfe
        logger.info("VariationalFreeEnergy conectado")
    
    # =========================================================================
    # OPERAÇÕES PRINCIPAIS
    # =========================================================================
    
    def trigger(self, 
                embedding: np.ndarray, 
                codes: Optional[np.ndarray] = None,
                intensity: float = 1.0) -> FieldState:
        """
        Triggera um conceito no campo.
        
        Isso:
        1. Projeta embedding na variedade
        2. Ativa o ponto
        3. Deforma a métrica
        4. Retorna estado do campo
        
        Args:
            embedding: Vetor 384D (ou dim configurada)
            codes: Códigos VQ-VAE [4] opcionais
            intensity: Força do trigger [0, 1+]
            
        Returns:
            FieldState com campos computados
        """
        # Se temos VQ-VAE e não temos codes, codifica
        if codes is None and self.vqvae is not None:
            try:
                import torch
                if not isinstance(embedding, torch.Tensor):
                    emb_tensor = torch.tensor(embedding, dtype=torch.float32)
                else:
                    emb_tensor = embedding
                _, codes_tensor = self.vqvae.encode(emb_tensor.unsqueeze(0))
                codes = codes_tensor.squeeze().cpu().numpy()
            except Exception as e:
                logger.warning(f"VQ-VAE encode falhou: {e}")
                codes = None
        
        # Embed na variedade
        point = self.manifold.embed(embedding, codes)
        
        # Gera ID
        point_id = f"trigger_{self.trigger_count}"
        self.trigger_count += 1
        
        # Adiciona e ativa
        self.manifold.add_point(point_id, point)
        self.manifold.activate_point(point_id, intensity)
        
        # Deforma métrica
        self.metric.deform_at_point(point)
        
        # Computa estado do campo
        state = self.field.compute_field()
        
        # Observa no Mycelial se conectado
        if self.mycelial is not None and codes is not None:
            try:
                self.mycelial.observe(codes)
            except Exception as e:
                logger.warning(f"Mycelial observe falhou: {e}")
        
        return state
    
    def propagate(self, 
                  field_state: Optional[FieldState] = None, 
                  steps: int = 5) -> List[FieldState]:
        """
        Propaga a dinâmica do campo.
        
        Args:
            field_state: Estado inicial (ou usa último)
            steps: Número de passos
            
        Returns:
            Lista de FieldStates ao longo do tempo
        """
        states = []
        
        if field_state is None:
            field_state = self.field.get_state()
        
        if field_state is None:
            return states
        
        states.append(field_state)
        
        for _ in range(steps):
            # Relaxa métrica
            self.metric.relax(rate=0.1)
            
            # Decai ativações
            self.manifold.decay_activations(rate=0.1)
            
            # Recomputa campo
            new_state = self.field.compute_field()
            states.append(new_state)
        
        return states
    
    def run_cycle(self, 
                  trigger_embedding: Optional[np.ndarray] = None) -> CycleState:
        """
        Roda um ciclo completo: Expansão → Configuração → Compressão.
        
        Args:
            trigger_embedding: Embedding que inicia o ciclo (opcional)
            
        Returns:
            CycleState com resultado
        """
        result = self.cycle.run_cycle(trigger_embedding)
        self.cycle_history.append(result)
        
        return result
    
    def crystallize(self) -> Dict[str, Any]:
        """
        Cristaliza o campo atual em estrutura de grafo.
        
        Transforma:
        - Atratores → Nós
        - Geodésicas → Arestas
        
        Se Mycelial conectado, incorpora no grafo Hebbiano.
        
        Returns:
            Dict com nodes e edges
        """
        state = self.field.get_state()
        
        if state is None or len(state.attractors) == 0:
            return {'nodes': [], 'edges': []}
        
        # Extrai atratores como nós
        nodes = []
        for i, attractor in enumerate(state.attractors):
            nodes.append({
                'id': f'attractor_{i}',
                'coordinates': attractor.tolist(),
                'free_energy': float(self.field.free_energy_at(attractor))
            })
        
        # Cria arestas baseadas em proximidade dos gradientes
        edges = []
        for i, node_a in enumerate(nodes):
            for j, node_b in enumerate(nodes):
                if i >= j:
                    continue
                
                coord_a = np.array(node_a['coordinates'])
                coord_b = np.array(node_b['coordinates'])
                
                # Distância Riemanniana aproximada
                dist = self.metric.distance(coord_a, coord_b)
                
                # Peso inversamente proporcional à distância
                weight = 1.0 / (1.0 + dist)
                
                if weight > 0.1:  # Threshold
                    edges.append({
                        'source': node_a['id'],
                        'target': node_b['id'],
                        'weight': float(weight)
                    })
        
        graph = {'nodes': nodes, 'edges': edges}
        
        # Se Mycelial conectado, incorpora
        if self.mycelial is not None:
            self._incorporate_to_mycelial(graph)
        
        return graph
    
    def _incorporate_to_mycelial(self, graph: Dict):
        """Incorpora grafo cristalizado no Mycelial."""
        # Para cada edge no grafo cristalizado,
        # cria/fortalece conexão no Mycelial
        
        # Isso requer mapear coordenadas → códigos VQ-VAE
        # Por agora, apenas loga
        logger.info(f"Cristalizado: {len(graph['nodes'])} nós, {len(graph['edges'])} arestas")
    
    # =========================================================================
    # QUERIES
    # =========================================================================
    
    def get_free_energy_at(self, point: np.ndarray) -> float:
        """Computa F em um ponto."""
        return self.field.free_energy_at(point)
    
    def get_gradient_at(self, point: np.ndarray) -> np.ndarray:
        """Computa ∇F em um ponto."""
        return self.field.gradient_at(point)
    
    def get_attractors(self) -> List[np.ndarray]:
        """Retorna atratores atuais."""
        state = self.field.get_state()
        return state.attractors if state else []
    
    def get_curvature_at(self, point: np.ndarray) -> float:
        """Computa curvatura em um ponto."""
        return self.metric.curvature_scalar_at(point)
    
    def stats(self) -> Dict[str, Any]:
        """Estatísticas completas do Campo."""
        return {
            'manifold': self.manifold.stats(),
            'metric': {
                'deformations': len(self.metric.deformations)
            },
            'field': self.field.stats(),
            'triggers': self.trigger_count,
            'cycles_completed': len(self.cycle_history),
            'connected': {
                'vqvae': self.vqvae is not None,
                'mycelial': self.mycelial is not None,
                'variational_fe': self.variational_fe is not None
            }
        }
    
    # =========================================================================
    # TEMPERATURA (exploration vs exploitation)
    # =========================================================================
    
    def set_temperature(self, T: float):
        """
        Ajusta temperatura.
        
        T alto: sistema explora (entropia domina)
        T baixo: sistema explota (energia domina)
        """
        self.field.set_temperature(T)
        self.config.temperature = T
    
    def anneal(self, 
               start_temp: float = 2.0, 
               end_temp: float = 0.1, 
               steps: int = 50) -> List[FieldState]:
        """
        Aplica simulated annealing.
        
        Returns:
            Estados ao longo do annealing
        """
        states = []
        
        for step in range(steps):
            # Schedule linear
            T = start_temp - (start_temp - end_temp) * (step / steps)
            self.set_temperature(T)
            
            # Evolui um passo
            self.manifold.decay_activations(rate=0.05)
            self.metric.relax(rate=0.05)
            
            state = self.field.compute_field()
            states.append(state)
        
        return states

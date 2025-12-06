"""
Predictive Coding Module for Alexandria
========================================

Implementação de Predictive Coding hierárquico para a rede Alexandria.
Baseado em: Karl Friston's Free Energy Principle e Predictive Processing.

Este módulo implementa a evolução natural do Meta-Hebbian:
- Em vez de propagar ATIVAÇÕES, propaga ERROS de predição
- Cada camada prediz o que vai receber e aprende com a diferença
- Mais eficiente (só transmite surpresa/novidade)
- Biologicamente plausível
- Preparação para Active Inference

Hierarquia de paradigmas:
    Hebbian → Meta-Hebbian → Predictive Coding → Active Inference → Free Energy
                                    ↑
                               VOCÊ ESTÁ AQUI

Referências:
- Rao & Ballard (1999) - Predictive coding in visual cortex
- Friston (2005) - A theory of cortical responses
- Whittington & Bogacz (2017) - Approximation of backprop

Autor: G (Alexandria Project)
Versão: 1.0
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import pickle
from pathlib import Path

# Import do Meta-Hebbian (assumindo que está no mesmo diretório)
try:
    from meta_hebbian import MetaHebbianPlasticity, PlasticityRule, create_meta_hebbian_system
except ImportError:
    # Fallback se rodando standalone
    MetaHebbianPlasticity = None


# =============================================================================
# CONFIGURAÇÃO E TIPOS
# =============================================================================

class PrecisionMode(Enum):
    """Como a precisão (confiança) é computada"""
    FIXED = "fixed"              # Precisão fixa
    LEARNED = "learned"          # Precisão aprendida por camada
    ADAPTIVE = "adaptive"        # Precisão adapta em runtime


@dataclass
class PredictiveCodingConfig:
    """Configuração do sistema de Predictive Coding"""
    
    # Arquitetura
    input_dim: int = 384                    # Dimensão do embedding (all-MiniLM)
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    code_dim: int = 32                      # Dimensão do código latente
    
    # Dinâmica
    num_iterations: int = 10                # Iterações de inferência por input
    inference_lr: float = 0.1               # Learning rate da inferência
    learning_lr: float = 0.01               # Learning rate do aprendizado
    
    # Precisão (inverse variance)
    precision_mode: PrecisionMode = PrecisionMode.ADAPTIVE
    base_precision: float = 1.0             # Precisão base
    precision_lr: float = 0.001             # LR para aprender precisão
    
    # Integração
    use_meta_hebbian: bool = True           # Usar Meta-Hebbian para pesos
    prediction_noise: float = 0.01          # Ruído nas predições (regularização)
    
    # Persistência
    save_path: str = "data/predictive_coding_state.pkl"


# =============================================================================
# CAMADA PREDITIVA
# =============================================================================

class PredictiveLayer:
    """
    Uma camada no modelo de Predictive Coding.
    
    Cada camada:
    1. Recebe predição top-down da camada acima
    2. Recebe input bottom-up da camada abaixo
    3. Computa erro de predição
    4. Atualiza representação para minimizar erro
    5. Propaga erro (não ativação!) para cima e para baixo
    
    Equações principais:
        prediction_error = input - prediction
        representation += lr * (precision * prediction_error - lateral_inhibition)
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        layer_id: int,
        config: PredictiveCodingConfig
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_id = layer_id
        self.config = config
        
        # Pesos
        self._init_weights()
        
        # Estado
        self.representation = np.zeros(output_dim)      # μ (mean)
        self.prediction_error = np.zeros(input_dim)     # ε (error)
        self.precision = np.ones(input_dim) * config.base_precision  # Π (precision)
        
        # Histórico
        self.error_history: List[float] = []
        self.precision_history: List[float] = []
        
    def _init_weights(self):
        """Inicializa pesos com Xavier/Glorot"""
        scale = np.sqrt(2.0 / (self.input_dim + self.output_dim))
        
        # W_pred: gera predição para camada abaixo
        self.W_pred = np.random.randn(self.input_dim, self.output_dim) * scale
        
        # W_err: processa erro da camada abaixo
        self.W_err = np.random.randn(self.output_dim, self.input_dim) * scale
        
        # Bias
        self.b_pred = np.zeros(self.input_dim)
        self.b_rep = np.zeros(self.output_dim)
        
    def predict(self, top_down_input: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Gera predição para a camada abaixo.
        
        prediction = W_pred @ representation + bias + noise
        """
        prediction = self.W_pred @ self.representation + self.b_pred
        
        # Adiciona ruído para regularização
        if self.config.prediction_noise > 0:
            prediction += np.random.randn(*prediction.shape) * self.config.prediction_noise
        
        # Non-linearity (ReLU suave)
        prediction = self._softplus(prediction)
        
        return prediction
    
    def compute_error(self, bottom_up_input: np.ndarray) -> np.ndarray:
        """
        Computa erro de predição.
        
        error = precision * (input - prediction)
        
        Precision-weighted: erros em dimensões "confiáveis" pesam mais.
        """
        prediction = self.predict()
        raw_error = bottom_up_input - prediction
        
        # Precision weighting
        self.prediction_error = self.precision * raw_error
        
        # Registra magnitude do erro
        self.error_history.append(float(np.mean(np.abs(self.prediction_error))))
        
        return self.prediction_error
    
    def update_representation(
        self,
        bottom_up_error: np.ndarray,
        top_down_prediction: Optional[np.ndarray] = None,
        iterations: Optional[int] = None
    ) -> np.ndarray:
        """
        Atualiza representação interna para minimizar erro.
        
        Isso é a "inferência" no Predictive Coding:
        - Não é feedforward pass
        - É um processo iterativo de settling
        - Minimiza energia livre variacional
        
        Δμ = lr * (W_err @ ε_below - ε_self)
        """
        iterations = iterations or self.config.num_iterations
        
        for _ in range(iterations):
            # Gradiente do erro bottom-up
            # W_err: (output_dim, input_dim), bottom_up_error: (input_dim,)
            # Resultado: (output_dim,) - mesmo tamanho que representation
            gradient_bottom = self.W_err @ bottom_up_error
            
            # Gradiente do erro top-down (se houver)
            if top_down_prediction is not None and len(top_down_prediction) == len(self.representation):
                error_top = self.representation - top_down_prediction
                gradient_top = -error_top
            else:
                gradient_top = np.zeros_like(self.representation)
            
            # Atualiza representação
            total_gradient = gradient_bottom + gradient_top
            self.representation += self.config.inference_lr * total_gradient
            
            # Regularização (mantém representação bounded)
            self.representation = np.clip(self.representation, -10, 10)
        
        return self.representation
    
    def learn(self, bottom_up_input: np.ndarray):
        """
        Atualiza pesos para melhorar predições futuras.
        
        ΔW_pred = lr * ε @ μ.T  (Hebbian no erro!)
        """
        # Computa erro atual
        error = self.compute_error(bottom_up_input)
        
        # Gradiente para W_pred
        # Minimiza ||input - W_pred @ representation||²
        dW_pred = np.outer(error, self.representation) * self.config.learning_lr
        self.W_pred += dW_pred
        
        # Gradiente para bias
        db_pred = error * self.config.learning_lr
        self.b_pred += db_pred
        
        # Atualiza precisão se modo adaptativo
        if self.config.precision_mode == PrecisionMode.ADAPTIVE:
            self._update_precision(error)
        
        return {
            'error_magnitude': float(np.mean(np.abs(error))),
            'weight_change': float(np.mean(np.abs(dW_pred))),
            'precision_mean': float(np.mean(self.precision))
        }
    
    def _update_precision(self, error: np.ndarray):
        """
        Atualiza precisão baseado na variância do erro.
        
        Precisão alta = erro consistentemente baixo = confiança alta
        Precisão baixa = erro variável = confiança baixa
        """
        # Erro quadrático como proxy para variância
        error_variance = error ** 2
        
        # Precisão é inverso da variância (com suavização)
        target_precision = 1.0 / (error_variance + 0.01)
        
        # Atualização suave
        self.precision += self.config.precision_lr * (target_precision - self.precision)
        
        # Clamp para estabilidade
        self.precision = np.clip(self.precision, 0.1, 10.0)
        
        self.precision_history.append(float(np.mean(self.precision)))
    
    def _softplus(self, x: np.ndarray) -> np.ndarray:
        """Softplus activation: log(1 + exp(x))"""
        return np.log1p(np.exp(np.clip(x, -20, 20)))
    
    def get_state(self) -> Dict[str, Any]:
        """Retorna estado completo da camada"""
        return {
            'representation': self.representation.copy(),
            'prediction_error': self.prediction_error.copy(),
            'precision': self.precision.copy(),
            'W_pred': self.W_pred.copy(),
            'W_err': self.W_err.copy(),
            'b_pred': self.b_pred.copy(),
            'error_history': self.error_history[-100:],
            'precision_history': self.precision_history[-100:]
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Restaura estado da camada"""
        self.representation = state['representation']
        self.prediction_error = state['prediction_error']
        self.precision = state['precision']
        self.W_pred = state['W_pred']
        self.W_err = state['W_err']
        self.b_pred = state['b_pred']
        self.error_history = state.get('error_history', [])
        self.precision_history = state.get('precision_history', [])


# =============================================================================
# REDE DE PREDICTIVE CODING
# =============================================================================

class PredictiveCodingNetwork:
    """
    Rede hierárquica de Predictive Coding.
    
    Arquitetura:
        Input (384D) → Layer 1 (256D) → Layer 2 (128D) → Layer 3 (64D) → Code (32D)
        
    Fluxo de informação:
        Bottom-up: erros de predição sobem
        Top-down: predições descem
        
    Diferença do feedforward tradicional:
        - Não é um pass único
        - É um processo iterativo de settling
        - Converge para estado de mínima energia livre
    """
    
    def __init__(self, config: Optional[PredictiveCodingConfig] = None):
        self.config = config or PredictiveCodingConfig()
        
        # Constrói camadas
        self.layers: List[PredictiveLayer] = []
        self._build_network()
        
        # Meta-Hebbian opcional
        self.meta_hebbian = None
        if self.config.use_meta_hebbian and MetaHebbianPlasticity:
            self.meta_hebbian = create_meta_hebbian_system(
                num_codes=self.config.code_dim,
                num_heads=4,
                load_existing=False
            )
        
        # Estatísticas
        self.total_observations = 0
        self.convergence_history: List[int] = []
        
    def _build_network(self):
        """Constrói stack de camadas preditivas"""
        dims = [self.config.input_dim] + self.config.hidden_dims + [self.config.code_dim]
        
        for i in range(len(dims) - 1):
            layer = PredictiveLayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                layer_id=i,
                config=self.config
            )
            self.layers.append(layer)
        
        print(f"🧠 Predictive Coding Network construída:")
        print(f"   Camadas: {' → '.join(str(d) for d in dims)}")
    
    def infer(
        self,
        input_data: np.ndarray,
        max_iterations: int = 50,
        convergence_threshold: float = 0.001
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Inferência: encontra representação latente que melhor explica o input.
        
        Processo iterativo:
        1. Propaga input bottom-up
        2. Gera predições top-down
        3. Computa erros
        4. Atualiza representações
        5. Repete até convergir
        
        Returns:
            code: Representação latente final
            stats: Estatísticas da inferência
        """
        # Inicializa representações
        for layer in self.layers:
            layer.representation = np.random.randn(layer.output_dim) * 0.1
        
        prev_total_error = float('inf')
        errors_over_time = []
        
        for iteration in range(max_iterations):
            total_error = 0
            
            # Bottom-up pass: computa erros
            current_input = input_data
            for layer in self.layers:
                error = layer.compute_error(current_input)
                total_error += np.mean(np.abs(error))
                current_input = layer.representation
            
            # Top-down pass: atualiza representações
            top_down_pred = None
            for layer in reversed(self.layers):
                # Cada camada usa seu próprio erro de predição
                bottom_up_err = layer.prediction_error
                
                layer.update_representation(bottom_up_err, top_down_pred, iterations=1)
                
                # Predição para a próxima camada (acima)
                if layer.layer_id > 0:
                    top_down_pred = layer.representation  # Passa representação como target
            
            errors_over_time.append(total_error)
            
            # Verifica convergência
            error_change = abs(prev_total_error - total_error)
            if error_change < convergence_threshold:
                self.convergence_history.append(iteration)
                break
            
            prev_total_error = total_error
        
        # Código latente é a representação da última camada
        code = self.layers[-1].representation.copy()
        
        self.total_observations += 1
        
        return code, {
            'iterations': iteration + 1,
            'final_error': total_error,
            'converged': error_change < convergence_threshold,
            'errors_over_time': errors_over_time
        }
    
    def learn_from_input(self, input_data: np.ndarray) -> Dict[str, Any]:
        """
        Aprendizado: atualiza pesos para melhorar predições.
        
        1. Primeiro faz inferência (encontra melhor representação)
        2. Depois atualiza pesos baseado nos erros
        """
        # Inferência
        code, infer_stats = self.infer(input_data)
        
        # Aprendizado camada por camada
        learn_stats = []
        current_input = input_data
        
        for layer in self.layers:
            stats = layer.learn(current_input)
            learn_stats.append(stats)
            current_input = layer.representation
        
        # Se usando Meta-Hebbian, evolui regras periodicamente
        meta_stats = None
        if self.meta_hebbian and self.total_observations % 50 == 0:
            # Usa erro como proxy para fitness (menor erro = maior fitness)
            fitness = 1.0 / (infer_stats['final_error'] + 0.01)
            meta_stats = self.meta_hebbian.evolve_rules([fitness])
        
        return {
            'inference': infer_stats,
            'learning': learn_stats,
            'meta_hebbian': meta_stats,
            'code': code
        }
    
    def encode(self, input_data: np.ndarray) -> np.ndarray:
        """Encoding rápido (inferência completa)"""
        code, _ = self.infer(input_data)
        return code
    
    def decode(self, code: np.ndarray) -> np.ndarray:
        """
        Decoding: gera predição do input a partir do código.
        
        Propaga top-down através das camadas.
        Cada camada gera predição que serve como representação da camada abaixo.
        """
        # Define representação da última camada
        self.layers[-1].representation = code.copy()
        
        # Propaga top-down
        for i in range(len(self.layers) - 1, 0, -1):
            # Camada i gera predição
            prediction = self.layers[i].predict()
            # Predição tem dimensão input_dim[i] = output_dim[i-1]
            # Então pode ser usada diretamente como representação de i-1
            self.layers[i - 1].representation = prediction
        
        # Predição final é a predição da primeira camada (384D)
        return self.layers[0].predict()
    
    def get_prediction_errors(self) -> List[np.ndarray]:
        """Retorna erros de predição de todas as camadas"""
        return [layer.prediction_error.copy() for layer in self.layers]
    
    def get_precisions(self) -> List[np.ndarray]:
        """Retorna precisões de todas as camadas"""
        return [layer.precision.copy() for layer in self.layers]
    
    # =========================================================================
    # ANÁLISE E DIAGNÓSTICO
    # =========================================================================
    
    def get_network_analysis(self) -> Dict[str, Any]:
        """Análise completa da rede"""
        layer_stats = []
        for i, layer in enumerate(self.layers):
            layer_stats.append({
                'layer_id': i,
                'dims': f"{layer.input_dim} → {layer.output_dim}",
                'mean_error': np.mean(layer.error_history[-10:]) if layer.error_history else 0,
                'mean_precision': np.mean(layer.precision),
                'precision_std': np.std(layer.precision),
                'representation_norm': np.linalg.norm(layer.representation),
                'weight_norm': np.linalg.norm(layer.W_pred)
            })
        
        # Convergência
        avg_iterations = np.mean(self.convergence_history[-100:]) if self.convergence_history else 0
        
        return {
            'layers': layer_stats,
            'total_observations': self.total_observations,
            'avg_convergence_iterations': avg_iterations,
            'meta_hebbian_active': self.meta_hebbian is not None,
            'interpretation': self._interpret_state()
        }
    
    def _interpret_state(self) -> str:
        """Interpretação legível do estado"""
        if not self.convergence_history:
            return "INITIALIZING: Ainda sem observações"
        
        avg_iterations = np.mean(self.convergence_history[-20:])
        avg_error = np.mean([
            np.mean(l.error_history[-10:]) if l.error_history else 1.0 
            for l in self.layers
        ])
        
        if avg_iterations < 5 and avg_error < 0.1:
            return "EFFICIENT: Convergência rápida, erros baixos"
        elif avg_iterations < 10:
            return "LEARNING: Convergência boa, ainda otimizando"
        elif avg_error > 0.5:
            return "STRUGGLING: Erros altos, modelo precisa mais treino"
        else:
            return "EXPLORING: Estado intermediário de aprendizado"
    
    # =========================================================================
    # PERSISTÊNCIA
    # =========================================================================
    
    def save_state(self, path: Optional[str] = None):
        """Salva estado completo da rede"""
        path = path or self.config.save_path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dims': self.config.hidden_dims,
                'code_dim': self.config.code_dim,
                'num_iterations': self.config.num_iterations,
                'precision_mode': self.config.precision_mode.value
            },
            'layers': [layer.get_state() for layer in self.layers],
            'total_observations': self.total_observations,
            'convergence_history': self.convergence_history[-1000:]
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        # Salva Meta-Hebbian separadamente se existir
        if self.meta_hebbian:
            self.meta_hebbian.save_state()
        
        return path
    
    def load_state(self, path: Optional[str] = None) -> bool:
        """Carrega estado salvo"""
        path = path or self.config.save_path
        
        if not Path(path).exists():
            return False
        
        try:
            with open(path, 'rb') as f:
                state = pickle.load(f)
            
            for i, layer_state in enumerate(state['layers']):
                if i < len(self.layers):
                    self.layers[i].set_state(layer_state)
            
            self.total_observations = state.get('total_observations', 0)
            self.convergence_history = state.get('convergence_history', [])
            
            # Carrega Meta-Hebbian
            if self.meta_hebbian:
                self.meta_hebbian.load_state()
            
            return True
        except Exception as e:
            print(f"Erro ao carregar Predictive Coding: {e}")
            return False


# =============================================================================
# INTEGRAÇÃO COM ALEXANDRIA
# =============================================================================

class PredictiveCodingAlexandriaIntegration:
    """
    Integração do Predictive Coding com o sistema Alexandria.
    
    Substitui/complementa o pipeline:
        Embedding → VQ-VAE → Mycelial
    
    Por:
        Embedding → PredictiveCoding → VQ-VAE → Mycelial
        
    Benefícios:
    - Representações mais compactas (só novidade)
    - Melhor generalização
    - Preparação para Active Inference
    """
    
    def __init__(
        self,
        pc_network: PredictiveCodingNetwork,
        vqvae_encoder: Optional[Any] = None,
        mycelial: Optional[Any] = None
    ):
        self.pc = pc_network
        self.vqvae = vqvae_encoder
        self.mycelial = mycelial
        
        # Buffer para batch processing
        self.embedding_buffer: List[np.ndarray] = []
        self.buffer_size = 32
        
    def process_embedding(
        self,
        embedding: np.ndarray,
        learn: bool = True
    ) -> Dict[str, Any]:
        """
        Processa um embedding através do pipeline PC.
        
        Args:
            embedding: Vetor 384D do sentence-transformer
            learn: Se True, atualiza pesos
            
        Returns:
            result: Dicionário com código, erros, etc.
        """
        if learn:
            result = self.pc.learn_from_input(embedding)
        else:
            code, infer_stats = self.pc.infer(embedding)
            result = {'code': code, 'inference': infer_stats}
        
        # Se tiver VQ-VAE, quantiza o código PC
        if self.vqvae is not None:
            pc_code = result['code']
            # Expande código PC para dimensão esperada pelo VQ-VAE se necessário
            if hasattr(self.vqvae, 'encode'):
                vq_indices = self.vqvae.encode(pc_code)
                result['vq_indices'] = vq_indices
        
        # Se tiver Mycelial, observa
        if self.mycelial is not None and 'vq_indices' in result:
            self.mycelial.observe(result['vq_indices'])
        
        return result
    
    def process_batch(
        self,
        embeddings: List[np.ndarray],
        learn: bool = True
    ) -> Dict[str, Any]:
        """Processa batch de embeddings"""
        results = []
        total_error = 0
        
        for emb in embeddings:
            result = self.process_embedding(emb, learn=learn)
            results.append(result)
            if 'inference' in result:
                total_error += result['inference'].get('final_error', 0)
        
        return {
            'batch_size': len(embeddings),
            'mean_error': total_error / len(embeddings) if embeddings else 0,
            'results': results
        }
    
    def get_surprise_signal(self, embedding: np.ndarray) -> float:
        """
        Computa "surpresa" do input.
        
        Surpresa alta = input muito diferente do esperado
        Surpresa baixa = input previsível
        
        Isso é útil para:
        - Detectar outliers
        - Priorizar aprendizado de coisas novas
        - Active Inference (próximo passo)
        """
        _, stats = self.pc.infer(embedding, max_iterations=5)
        
        # Surpresa é proporcional ao erro de predição
        surprise = stats['final_error']
        
        return surprise
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Estatísticas da integração"""
        return {
            'pc_stats': self.pc.get_network_analysis(),
            'has_vqvae': self.vqvae is not None,
            'has_mycelial': self.mycelial is not None,
            'buffer_size': len(self.embedding_buffer)
        }


# =============================================================================
# ACTIVE INFERENCE PREVIEW (próximo passo)
# =============================================================================

class ActiveInferencePreview:
    """
    Preview de Active Inference.
    
    Active Inference = Predictive Coding + AÇÃO
    
    O sistema não só prediz passivamente, mas ATUA no mundo
    para confirmar suas predições (ou reduzir incerteza).
    
    Para Alexandria, isso significaria:
    - Sistema que busca ativamente papers para preencher gaps
    - Queries que o sistema gera sozinho
    - Exploração autônoma do espaço de conhecimento
    
    NOTA: Esta é uma preview. Implementação completa requer
    definir espaço de ações (queries, navegação, etc.)
    """
    
    def __init__(self, pc_network: PredictiveCodingNetwork):
        self.pc = pc_network
        self.action_history: List[Dict] = []
        
    def compute_expected_free_energy(
        self,
        possible_actions: List[str],
        current_state: np.ndarray
    ) -> List[Tuple[str, float]]:
        """
        Computa energia livre esperada para cada ação possível.
        
        G = E[log P(o|π) - log Q(o|π)] + E[H(o|s,π)]
        
        Simplificado:
        G ≈ uncertainty_reduction - information_gain
        
        Ação ótima = minimiza G
        """
        action_scores = []
        
        for action in possible_actions:
            # Simula efeito da ação (simplificado)
            # Em implementação real, usaria modelo do mundo
            
            # Proxy: ações que reduzem incerteza nas camadas
            uncertainty = np.mean([np.mean(l.precision) for l in self.pc.layers])
            
            # Score: menor G é melhor
            G = -uncertainty  # Simplificação
            
            action_scores.append((action, G))
        
        # Ordena por G (menor é melhor)
        action_scores.sort(key=lambda x: x[1])
        
        return action_scores
    
    def suggest_next_action(self) -> Dict[str, Any]:
        """
        Sugere próxima ação baseado em Active Inference.
        
        Para Alexandria:
        - Se incerteza alta em cluster X → buscar papers sobre X
        - Se conexão fraca entre A e B → buscar papers que conectam
        """
        # Analisa estado atual
        analysis = self.pc.get_network_analysis()
        
        # Identifica áreas de alta incerteza (baixa precisão)
        low_precision_layers = [
            l for l in analysis['layers'] 
            if l['mean_precision'] < 1.0
        ]
        
        if low_precision_layers:
            return {
                'action_type': 'EXPLORE',
                'target': f"Layer {low_precision_layers[0]['layer_id']}",
                'reason': 'Alta incerteza detectada',
                'priority': 1.0 - low_precision_layers[0]['mean_precision']
            }
        else:
            return {
                'action_type': 'CONSOLIDATE',
                'target': 'All layers',
                'reason': 'Sistema estável, consolidar conhecimento',
                'priority': 0.3
            }


# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def create_predictive_coding_system(
    input_dim: int = 384,
    hidden_dims: Optional[List[int]] = None,
    code_dim: int = 32,
    load_existing: bool = True,
    use_meta_hebbian: bool = True
) -> PredictiveCodingNetwork:
    """
    Factory function para criar sistema de Predictive Coding.
    """
    config = PredictiveCodingConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims or [256, 128, 64],
        code_dim=code_dim,
        use_meta_hebbian=use_meta_hebbian
    )
    
    network = PredictiveCodingNetwork(config)
    
    if load_existing:
        loaded = network.load_state()
        if loaded:
            print(f"✅ Predictive Coding carregado: {network.total_observations} observações")
        else:
            print("🌱 Predictive Coding inicializado fresh")
    
    return network


def integrate_with_alexandria(
    pc_network: PredictiveCodingNetwork,
    vqvae=None,
    mycelial=None
) -> PredictiveCodingAlexandriaIntegration:
    """
    Integra Predictive Coding com componentes Alexandria.
    """
    return PredictiveCodingAlexandriaIntegration(pc_network, vqvae, mycelial)


# =============================================================================
# EXEMPLO DE USO E TESTES
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("PREDICTIVE CODING - ALEXANDRIA")
    print("=" * 70)
    
    # Criar rede
    pc = create_predictive_coding_system(
        input_dim=384,
        hidden_dims=[256, 128, 64],
        code_dim=32,
        use_meta_hebbian=True
    )
    
    # Simular alguns inputs
    print("\n🔄 SIMULANDO APRENDIZADO...")
    
    for i in range(20):
        # Embedding fake (normalmente viria do sentence-transformer)
        fake_embedding = np.random.randn(384)
        fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)
        
        # Aprende
        result = pc.learn_from_input(fake_embedding)
        
        if i % 5 == 0:
            print(f"   Obs {i+1}: erro={result['inference']['final_error']:.4f}, "
                  f"iters={result['inference']['iterations']}")
    
    # Análise
    print("\n📊 ANÁLISE DA REDE:")
    analysis = pc.get_network_analysis()
    
    for layer in analysis['layers']:
        print(f"\n   Layer {layer['layer_id']} ({layer['dims']}):")
        print(f"      Erro médio: {layer['mean_error']:.4f}")
        print(f"      Precisão: {layer['mean_precision']:.4f} ± {layer['precision_std']:.4f}")
    
    print(f"\n🎯 ESTADO: {analysis['interpretation']}")
    print(f"   Observações: {analysis['total_observations']}")
    print(f"   Convergência média: {analysis['avg_convergence_iterations']:.1f} iterações")
    
    # Teste de encoding/decoding
    print("\n🔄 TESTE DE RECONSTRUCTION:")
    test_input = np.random.randn(384)
    test_input = test_input / np.linalg.norm(test_input)
    
    code = pc.encode(test_input)
    reconstruction = pc.decode(code)
    
    recon_error = np.mean((test_input - reconstruction) ** 2)
    print(f"   Erro de reconstrução: {recon_error:.4f}")
    print(f"   Compressão: 384D → {len(code)}D ({len(code)/384*100:.1f}%)")
    
    # Active Inference preview
    print("\n🔮 ACTIVE INFERENCE PREVIEW:")
    ai_preview = ActiveInferencePreview(pc)
    suggestion = ai_preview.suggest_next_action()
    print(f"   Ação sugerida: {suggestion['action_type']}")
    print(f"   Target: {suggestion['target']}")
    print(f"   Razão: {suggestion['reason']}")
    
    # Salvar
    save_path = pc.save_state()
    print(f"\n💾 Estado salvo em: {save_path}")
    
    print("\n" + "=" * 70)
    print("✅ PREDICTIVE CODING PRONTO PARA INTEGRAÇÃO")
    print("=" * 70)
    
    print("""
    
ARQUITETURA COMPLETA:
=====================

    Input (embedding 384D)
           ↓
    ┌──────────────────────────────────────┐
    │     PREDICTIVE CODING NETWORK        │
    │                                      │
    │   Layer 1: 384 → 256                │
    │      ↓ erro ↑ predição              │
    │   Layer 2: 256 → 128                │
    │      ↓ erro ↑ predição              │
    │   Layer 3: 128 → 64                 │
    │      ↓ erro ↑ predição              │
    │   Layer 4: 64 → 32                  │
    │                                      │
    │   + Meta-Hebbian (regras adaptivas)  │
    └──────────────────────────────────────┘
           ↓
    Code (32D) → VQ-VAE → Mycelial
    

CAMINHO DE EVOLUÇÃO:
====================

    ✅ Hebbian (base)
    ✅ Meta-Hebbian (regras aprendidas)
    ✅ Predictive Coding (este arquivo)
    ⬜ Active Inference (preview incluído)
    ⬜ Free Energy completo (futuro)
    

INTEGRAÇÃO:
===========

    from predictive_coding import create_predictive_coding_system, integrate_with_alexandria
    
    # Criar sistema
    pc = create_predictive_coding_system()
    
    # Integrar com Alexandria
    integration = integrate_with_alexandria(pc, vqvae, mycelial)
    
    # Processar embeddings
    for embedding in embeddings:
        result = integration.process_embedding(embedding)
        
    # Checar surpresa
    surprise = integration.get_surprise_signal(new_embedding)
    if surprise > threshold:
        print("Input muito novo/surpreendente!")
    
    """)

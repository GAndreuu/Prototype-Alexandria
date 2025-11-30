"""
MONOLITH V11 - VISION ENCODER REAL
Convergência Ontológica com Otimização Termodinâmica Adaptativa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Dict, Optional, List
import math
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# 1. CONFIGURAÇÃO TERMODINÂMICA ADAPTATIVA
# ============================================================================

class AdaptiveThermodynamics:
    """Sistema de controle termodinâmico adaptativo com β-scheduler."""
    
    def __init__(self):
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.BATCH_SIZE = 64
        self.EPOCHS = 10
        self.NOISE_FACTOR = 0.6
        
        # β-Scheduler Parameters
        self.BETA_MIN = 1e-4
        self.BETA_MAX = 1e-1
        self.BETA_WARMUP_EPOCHS = 3
        self.ACCURACY_THRESHOLD = 0.85
        
        # Hierarchical VQ Parameters
        self.COARSE_DIM = 128
        self.FINE_DIM = 64
        self.COARSE_CODEBOOK = 16   # Categorias abstratas
        self.FINE_CODEBOOK = 256    # Detalhes finos
        
        # Disentanglement Parameters
        self.GAMMA = 10.0  # Factor-VAE penalty
        self.TC_WEIGHT = 1.0  # Total Correlation weight
        
        # Performance tracking
        self.beta_history = []
        self.loss_momentum = 0.0
        
    def compute_beta(self, epoch: int, accuracy: float, loss_delta: float = 0) -> float:
        """β-Scheduler dinâmico baseado em performance com momentum."""
        # Fase 1: Warmup (aprendizado básico)
        if epoch < self.BETA_WARMUP_EPOCHS:
            beta = self.BETA_MIN
            
        # Fase 2: Adaptação baseada em accuracy E momentum da loss
        else:
            # Atualiza momentum (média móvel da mudança de loss)
            self.loss_momentum = 0.9 * self.loss_momentum + 0.1 * loss_delta
            
            if accuracy < self.ACCURACY_THRESHOLD:
                # Sistema ainda aprendendo - baixa pressão
                beta = self.BETA_MIN
            else:
                # Sistema competente - aumenta pressão para compressão máxima
                progress = min((accuracy - self.ACCURACY_THRESHOLD) / 0.15, 1.0)
                
                # Ajusta baseado no momentum - se loss parou de melhorar, aumenta pressão
                if abs(self.loss_momentum) < 1e-4:  # Loss estagnada
                    progress = min(progress * 1.5, 1.0)
                
                beta = self.BETA_MIN * (1 - progress) + self.BETA_MAX * progress
                
                # Smooth transition usando histórico
                if self.beta_history:
                    beta = 0.7 * beta + 0.3 * self.beta_history[-1]
        
        self.beta_history.append(beta)
        return beta
    
    def entropy_pressure(self, epoch: int) -> float:
        """Calcula a pressão entrópica do ambiente com componentes caóticos."""
        # Simula variações ambientais (ciclos dia/noite termodinâmicos)
        cycle = math.sin(2 * math.pi * epoch / self.EPOCHS)
        
        # Adiciona componente caótico para simular perturbações reais
        chaos = np.random.normal(0, 0.05)  # 5% de variação aleatória
        
        return self.NOISE_FACTOR * (1.0 + 0.3 * cycle + chaos)

config = AdaptiveThermodynamics()

# ============================================================================
# 2. HIERARCHICAL VECTOR QUANTIZATION
# ============================================================================

class HierarchicalVQ(nn.Module):
    """Vector Quantization Hierárquico de 2 níveis com fluxo bidirecional."""
    
    def __init__(self, coarse_dim, fine_dim, coarse_book, fine_book):
        super().__init__()
        
        # Nível Coarse: Abstração categórica
        self.coarse_vq = VectorQuantizerWithStats(coarse_book, coarse_dim, 0.25)
        
        # Nível Fine: Detalhes residuais
        self.fine_vq = VectorQuantizerWithStats(fine_book, fine_dim, 0.25)
        
        # Projeções entre níveis (bidirecional)
        self.coarse_to_fine = nn.Sequential(
            nn.Linear(coarse_dim, fine_dim),
            nn.LayerNorm(fine_dim),
            nn.LeakyReLU(0.1)
        )
        
        self.fine_to_coarse = nn.Sequential(
            nn.Linear(fine_dim, coarse_dim),
            nn.LayerNorm(coarse_dim),
            nn.LeakyReLU(0.1)
        )
        
        # Gating mechanism para controlar influência
        self.coarse_gate = nn.Parameter(torch.tensor(0.5))
        self.fine_gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, z_continuous: torch.Tensor) -> Dict:
        """Quantização hierárquica com métricas e fluxo bidirecional."""
        
        # Split adaptativo do espaço latente
        split_point = z_continuous.size(1) // 2
        z_coarse = z_continuous[:, :split_point]
        z_fine = z_continuous[:, split_point:]
        
        # Nível 1: Quantização Coarse
        z_coarse_q, coarse_loss, coarse_stats = self.coarse_vq(z_coarse)
        
        # Influência coarse → fine
        coarse_influence = self.coarse_to_fine(z_coarse_q.detach())
        
        # Residual guiado pelo coarse com gate
        gate_c = torch.sigmoid(self.coarse_gate)
        residual = z_fine - gate_c * coarse_influence
        
        # Nível 2: Quantização Fine do residual
        z_fine_q, fine_loss, fine_stats = self.fine_vq(residual)
        
        # Feedback: fine → coarse (pequena correção)
        fine_feedback = self.fine_to_coarse(z_fine_q.detach())
        gate_f = torch.sigmoid(self.fine_gate) 
        z_coarse_corrected = z_coarse_q + 0.1 * gate_f * fine_feedback
        
        # Reconstrução completa com correção
        z_fine_reconstructed = z_fine_q + gate_c * coarse_influence
        z_hierarchical = torch.cat([z_coarse_corrected, z_fine_reconstructed], dim=1)
        
        # Information flow metrics
        with torch.no_grad():
            coarse_info = torch.var(z_coarse_q).item()
            fine_info = torch.var(z_fine_q).item()
            total_info = torch.var(z_hierarchical).item()
            compression_ratio = total_info / (coarse_info + fine_info + 1e-8)
        
        return {
            'z_quantized': z_hierarchical,
            'vq_loss': coarse_loss + fine_loss,
            'coarse_perplexity': coarse_stats['perplexity'],
            'fine_perplexity': fine_stats['perplexity'],
            'coarse_usage': coarse_stats['usage'],
            'fine_usage': fine_stats['usage'],
            'coarse_active': coarse_stats.get('active', coarse_stats['usage']),
            'fine_active': fine_stats.get('active', fine_stats['usage']),
            'compression_ratio': compression_ratio,
            'coarse_gate': gate_c.item(),
            'fine_gate': gate_f.item()
        }

class VectorQuantizerWithStats(nn.Module):
    """VQ com estatísticas de uso (perplexity e codebook usage)."""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        
        # Inicialização melhorada usando Xavier/He
        nn.init.xavier_uniform_(self._embedding.weight)
        
        self._commitment_cost = commitment_cost
        
        # Estatísticas de uso do codebook
        self.register_buffer('_ema_cluster_size', torch.ones(num_embeddings))
        self.register_buffer('_ema_weight', self._embedding.weight.clone())
        self._decay = 0.99
        self._epsilon = 1e-5
        
        # Tracking adicional
        self.register_buffer('usage_count', torch.zeros(num_embeddings))
        self.updates = 0
        
    def forward(self, inputs):
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Distâncias L2 com estabilidade numérica
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        # Encoding com temperature para exploração inicial
        temperature = max(1.0 - self.updates / 1000, 0.1)  # Decai de 1.0 para 0.1
        encoding_indices = torch.argmin(distances / temperature, dim=1).unsqueeze(1)
        
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantização
        quantized = torch.matmul(encodings, self._embedding.weight).view(inputs.shape)
        
        # Update embeddings usando EMA durante treino
        if self.training:
            self.updates += 1
            encoding_sum = encodings.sum(0)
            
            # EMA update
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * encoding_sum
                                     
            # Laplace smoothing para estabilidade
            n = self._ema_cluster_size.sum()
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon) /
                (n + self._num_embeddings * self._epsilon) * n
            )
            
            # Update weights
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_weight = self._ema_weight * self._decay + (1 - self._decay) * dw
            self._embedding.weight = self._ema_weight / self._ema_cluster_size.unsqueeze(1)
            
            # Track usage
            self.usage_count += encoding_sum
        
        # Losses com gradient balancing
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        
        # Estatísticas melhoradas
        with torch.no_grad():
            # Perplexity (medida de uso efetivo do codebook)
            avg_probs = self._ema_cluster_size / self._ema_cluster_size.sum()
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
            
            # Usage rate (códigos usados pelo menos uma vez)
            usage = (self.usage_count > 0).float().mean()
            
            # Active codes (usado recentemente)
            active = (self._ema_cluster_size > 0.001).float().mean()
        
        return quantized, loss, {
            'perplexity': perplexity, 
            'usage': usage,
            'active': active,
            'temperature': temperature
        }

# ============================================================================
# 3. DISENTANGLEMENT METRICS E LOSSES
# ============================================================================

class DisentanglementLoss(nn.Module):
    """Factor-VAE loss para desemaranhamento."""
    
    def __init__(self, gamma=10.0):
        super().__init__()
        self.gamma = gamma
        
    def compute_total_correlation(self, z, mu, logvar):
        """Calcula a Total Correlation (TC) para penalizar dependências."""
        
        # Amostra do posterior q(z)
        batch_size = z.size(0)
        
        # log q(z) - média sobre o batch
        log_qz = self._log_density_gaussian(z, mu, logvar)
        
        # log q(z_j) para cada dimensão j
        log_qz_marginals = 0
        for j in range(z.size(1)):
            mu_j = mu[:, j:j+1]
            logvar_j = logvar[:, j:j+1]
            z_j = z[:, j:j+1]
            log_qz_marginals += self._log_density_gaussian(z_j, mu_j, logvar_j)
        
        # TC = KL(q(z) || prod_j q(z_j))
        tc = (log_qz - log_qz_marginals).mean()
        return tc
    
    def _log_density_gaussian(self, z, mu, logvar):
        """Log densidade de uma gaussiana."""
        normalization = -0.5 * (math.log(2 * math.pi) + logvar)
        inv_var = torch.exp(-logvar)
        log_density = normalization - 0.5 * (z - mu)**2 * inv_var
        return log_density.sum(dim=1)
    
    def forward(self, z, mu, logvar):
        """Calcula a perda de desemaranhamento."""
        tc = self.compute_total_correlation(z, mu, logvar)
        return self.gamma * tc

# ============================================================================
# 4. RENORMALIZATION HIERÁRQUICA
# ============================================================================

class AdaptiveRenormalizationBlock(nn.Module):
    """Bloco de renormalização com skip connections adaptativas e self-attention."""
    
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        # Convoluções com diferentes kernel sizes para multi-scale
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, 1)  # Pointwise
        
        # Batch normalization com momentum adaptativo
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.1)
        
        # Pooling adaptativo
        self.pool = nn.AvgPool2d(scale_factor) if scale_factor > 1 else nn.Identity()
        
        # Skip connection adaptativa
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        # Gate aprendível para controlar fluxo de informação
        self.gate = nn.Parameter(torch.ones(1) * 0.5)
        
        # Self-attention leve para capturar dependências
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 8, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Dropout espacial para regularização
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        identity = self.skip(x)
        
        # Primeira convolução com residual
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.1)
        
        # Segunda convolução
        out = self.bn2(self.conv2(out))
        
        # Attention mechanism
        att = self.attention(out)
        out = out * att
        
        # Conexão 1x1 para mixar canais
        out = out + 0.1 * self.conv1x1(out)  # Pequena contribuição
        
        # Skip connection com gate adaptativo (clamped para estabilidade)
        gate_value = torch.sigmoid(self.gate)
        out = out * gate_value + identity * (1 - gate_value)
        
        # Ativação e pooling
        out = F.leaky_relu(out, 0.1)
        
        # Dropout durante treino
        if self.training:
            out = self.dropout(out)
            
        out = self.pool(out)
        
        return out

# ============================================================================
# 5. MONOLITH V11 VISION ENCODER
# ============================================================================

class MonolithV11VisionEncoder(nn.Module):
    """Sistema completo de compressão semântica hierárquica."""
    
    def __init__(self):
        super().__init__()
        
        # Feature Extraction Hierarchy
        self.features = nn.ModuleList([
            AdaptiveRenormalizationBlock(1, 32, 2),    # 28->14
            AdaptiveRenormalizationBlock(32, 64, 2),   # 14->7
            AdaptiveRenormalizationBlock(64, 128, 1),  # 7->7 (sem pooling)
        ])
        
        self.flatten_dim = 128 * 7 * 7
        
        # Variational Bottleneck
        latent_dim = config.COARSE_DIM + config.FINE_DIM
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        
        # Hierarchical Vector Quantization
        self.hvq = HierarchicalVQ(
            config.COARSE_DIM, config.FINE_DIM,
            config.COARSE_CODEBOOK, config.FINE_CODEBOOK
        )
        
        # Disentanglement Loss
        self.disentangle_loss = DisentanglementLoss(config.GAMMA)
        
        # Task Heads
        self.classifier = nn.Linear(latent_dim, 10)
        
        # Semantic Decoder
        self.decoder_input = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder_net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 7->14
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 14->28
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 1, 3, 1, 1),    # 28->28
            nn.Sigmoid()
        )
        
    def encode(self, x):
        """Codificação hierárquica com métricas."""
        # Extract features progressively
        for block in self.features:
            x = block(x)
        
        x_flat = x.view(x.size(0), -1)
        
        # Variational encoding
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_continuous = mu + eps * std
        
        return z_continuous, mu, logvar
    
    def decode(self, z):
        """Decodificação a partir do espaço latente."""
        z_spatial = self.decoder_input(z).view(-1, 128, 7, 7)
        reconstruction = self.decoder_net(z_spatial)
        return reconstruction
    
    def forward(self, x):
        # Encode
        z_continuous, mu, logvar = self.encode(x)
        
        # Hierarchical Quantization
        hvq_output = self.hvq(z_continuous)
        z_quantized = hvq_output['z_quantized']
        
        # Classify
        logits = self.classifier(z_quantized)
        
        # Decode
        reconstruction = self.decode(z_quantized)
        
        # Disentanglement
        disentangle_loss = self.disentangle_loss(z_continuous, mu, logvar)
        
        return {
            'logits': logits,
            'reconstruction': reconstruction,
            'mu': mu,
            'logvar': logvar,
            'vq_loss': hvq_output['vq_loss'],
            'disentangle_loss': disentangle_loss,
            'metrics': {
                'coarse_perplexity': hvq_output['coarse_perplexity'],
                'fine_perplexity': hvq_output['fine_perplexity'],
                'coarse_usage': hvq_output['coarse_usage'],
                'fine_usage': hvq_output['fine_usage']
            }
        }
    
    def get_image_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extrai embeddings de imagem para integração com SFS.
        
        Args:
            x: Tensor de imagem (batch_size x 3 x 224 x 224)
            
        Returns:
            Tensor de embeddings 384D normalizados
        """
        with torch.no_grad():
            z_continuous, mu, logvar = self.encode(x)
            
            # Usar o espaço latente contínuo para embeddings
            # Adicionar redução para 384D se necessário
            if z_continuous.size(1) != 384:
                # Projetar para 384D se necessário
                reduction_layer = nn.Linear(z_continuous.size(1), 384).to(x.device)
                z_384d = reduction_layer(z_continuous)
            else:
                z_384d = z_continuous
            
            # Normalizar para compatibilidade
            z_384d = F.normalize(z_384d, dim=1)
            
        return z_384d.cpu()

# ============================================================================
# 6. PROCESSAMENTO SIMPLIFICADO PARA SFS
# ============================================================================

class V11VisionEncoderSimplified:
    """Interface simplificada para integração com SFS"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        
        # Transformações para imagens
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),  # Para V11 (MNIST-like)
            transforms.Grayscale(1),      # Converter para grayscale
            transforms.ToTensor(),        # Para tensor [0,1]
        ])
        
    def load_model(self) -> bool:
        """Carrega o modelo V11"""
        try:
            if self.model_loaded:
                return True
            
            print("Carregando MonolithV11VisionEncoder...")
            
            # Criar e carregar o modelo
            self.model = MonolithV11VisionEncoder()
            
            # Aplicar FP16 se GPU disponível
            if torch.cuda.is_available():
                self.model = self.model.half()
                print("Modelo convertido para FP16")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            
            print("MonolithV11VisionEncoder carregado com sucesso!")
            return True
            
        except Exception as e:
            print(f"Erro ao carregar V11 Vision Encoder: {e}")
            return False
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Codifica uma imagem e retorna vetor 384D
        
        Args:
            image_path: Caminho para a imagem
            
        Returns:
            Vetor 384D numpy ou None se erro
        """
        try:
            if not self.model_loaded:
                if not self.load_model():
                    return None
            
            # Carregar e processar imagem
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image)  # [1, 28, 28]
            
            # Adicionar batch dimension
            image_batch = image_tensor.unsqueeze(0).to(self.device)  # [1, 1, 28, 28]
            
            # Gerar embeddings
            with torch.no_grad():
                embeddings = self.model.get_image_embeddings(image_batch)
            
            return embeddings.squeeze().numpy()
            
        except Exception as e:
            print(f"Erro ao processar imagem {image_path}: {e}")
            return None
    
    def batch_encode_images(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """Codifica múltiplas imagens"""
        results = []
        for path in image_paths:
            results.append(self.encode_image(path))
        return results

# ============================================================================
# 7. UTILITÁRIOS
# ============================================================================

def create_v11_encoder() -> V11VisionEncoderSimplified:
    """Factory function para criar encoder V11"""
    return V11VisionEncoderSimplified()

def test_v11_encoder():
    """Teste simples do encoder V11"""
    print("="*50)
    print("TESTE V11 VISION ENCODER")
    print("="*50)
    
    encoder = create_v11_encoder()
    
    # Teste com imagem fictícia (28x28 grayscale)
    test_image = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    from PIL import Image
    test_img = Image.fromarray(test_image, mode='L')
    test_img.save('test_image.png')
    
    result = encoder.encode_image('test_image.png')
    
    if result is not None:
        print(f"✅ Embedding gerado: shape {result.shape}")
        print(f"Primeiros 5 valores: {result[:5]}")
        return True
    else:
        print("❌ Falha ao gerar embedding")
        return False

if __name__ == "__main__":
    test_v11_encoder()

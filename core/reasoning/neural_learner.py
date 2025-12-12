import torch
import torch.optim as optim
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from core.reasoning.vqvae.model import MonolithV13
from core.reasoning.vqvae.loss import compute_orthogonal_loss, compute_vq_commitment_loss

logger = logging.getLogger(__name__)

class V2Learner:
    """
    Prototype Alexandria - Neural Learner (V2 Adapter)
    Integration with VQ-VAE Monolith V13

    This module adapts the V2 neural core to the V1 logic system, enabling
    the "Self-Feeding Cycle" where logical hypotheses become neural weights.

    Autor: Prototype Alexandria Team
    Data: 2025-11-28
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.model_dir = Path("data")
        self.model_path = self.model_dir / "monolith_v13_trained.pth"
        if model_path:
            self.model_path = Path(model_path)
        
        self.model = MonolithV13().to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.is_loaded = False
        
        self.history = []
        self.history_path = self.model_path.parent / "history.json"
        
        self._load_model()
        self._load_history()
        
    def _load_history(self):
        """Carrega histórico de aprendizado"""
        if self.history_path.exists():
            try:
                import json
                with open(self.history_path, 'r') as f:
                    self.history = json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar histórico: {e}")
        
    def _load_model(self):
        """Carrega pesos do modelo se existirem"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.is_loaded = True
                logger.info(f"✅ V2 Model carregado de {self.model_path}")
            except Exception as e:
                logger.error(f"❌ Erro ao carregar modelo V2: {e}")
                # Inicializa do zero se falhar
                self.is_loaded = True 
        else:
            logger.info("⚠️ Modelo V2 não encontrado. Iniciando do zero.")
            self.is_loaded = True

    def save_model(self):
        """Salva o estado atual do modelo"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, self.model_path)
            logger.info(f"💾 Modelo V2 salvo em {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Erro ao salvar modelo V2: {e}")
            
    def _save_history(self):
        """Salva histórico de aprendizado"""
        try:
            import json
            with open(self.history_path, 'w') as f:
                json.dump(self.history, f)
        except Exception as e:
            logger.error(f"Erro ao salvar histórico: {e}")

    def learn(self, vectors: List[List[float]]) -> Dict[str, float]:
        """
        Realiza um passo de treinamento (learning step) com os vetores fornecidos.
        
        Args:
            vectors: Lista de vetores (embeddings) para aprender.
            
        Returns:
            Dict com métricas de loss.
        """
        if not vectors:
            return {}
            
        self.model.train()
        
        # Converter para tensor
        data = torch.tensor(vectors, dtype=torch.float32).to(self.device)
        
        # Forward pass
        self.optimizer.zero_grad()
        output = self.model(data)
        
        # Calcular Loss
        # 1. Reconstrução (MSE)
        recon_loss = torch.nn.functional.mse_loss(output['reconstructed'], data)
        
        # 2. VQ Commitment Loss
        vq_loss = compute_vq_commitment_loss(output['z_e'], output['z_q'])
        
        # 3. Orthogonal Loss (para diversidade das heads)
        ortho_loss = compute_orthogonal_loss(self.model.quantizer)
        
        # Loss Total
        total_loss = recon_loss + vq_loss + 0.1 * ortho_loss
        
        # Backward pass
        total_loss.backward()
        self.optimizer.step()
        
        # Adicionar ao histórico
        from datetime import datetime
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "total_loss": total_loss.item(),
                "recon_loss": recon_loss.item(),
                "vq_loss": vq_loss.item(),
                "ortho_loss": ortho_loss.item()
            }
        })
        self._save_history()
        
        return {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "vq_loss": vq_loss.item(),
            "ortho_loss": ortho_loss.item()
        }

    def encode(self, vectors: List[List[float]]) -> np.ndarray:
        """Retorna a representação latente (quantizada) dos vetores"""
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(vectors, dtype=torch.float32).to(self.device)
            output = self.model(data)
            return output['z_q'].cpu().numpy()

    def decode(self, latents: List[List[float]]) -> np.ndarray:
        """Reconstrói vetores a partir do espaço latente"""
        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(latents, dtype=torch.float32).to(self.device)
            out = self.model.decoder(data)
            return out.cpu().numpy()

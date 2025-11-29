import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class QuaternionUtils:
    """
    Utility class for quaternion operations.
    Quaternions are represented as [w, x, y, z] where w is the real part.
    """
    
    @staticmethod
    def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
        return F.normalize(q, p=2, dim=-1, eps=1e-8)
    
    @staticmethod
    def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
        """
        Convert quaternion to 3x3 rotation matrix.
        
        Args:
            q: (..., 4) quaternion [w, x, y, z]
            
        Returns:
            R: (..., 3, 3) rotation matrix
        """
        # Normalize first
        q = QuaternionUtils.normalize_quaternion(q)
        
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        
        # Compute rotation matrix elements
        R = torch.stack([
            torch.stack([1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
            torch.stack([2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)], dim=-1),
            torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)], dim=-1)
        ], dim=-2)
        
        return R


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(1)]


class PrimitiveTransformerQuaternion(nn.Module):
    """
    Transformer that predicts primitive parameters (SRT + class) from point cloud features.
    
    Uses quaternion rotation representation (8D: μ and σ for 4D quaternion).
    Quaternions are normalized to enforce unit norm constraint.
    """
    
    def __init__(
        self,
        n_primitives: int = 8,
        point_feature_dim: int = 256,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        n_classes: int = 3,  # sphere, cylinder, cuboid
    ):
        super().__init__()
        
        self.n_primitives = n_primitives
        self.d_model = d_model
        self.n_classes = n_classes
        
        # Project point features to model dimension
        self.point_feature_proj = nn.Linear(point_feature_dim, d_model)
        
        # Learnable query embeddings for primitives
        self.primitive_queries = nn.Embedding(n_primitives, d_model)
        
        # SOS token
        self.sos_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.query_pos_encoding = PositionalEncoding(d_model, max_len=n_primitives+1)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output heads
        self._build_output_heads()
        
        self._init_weights()
    
    def _build_output_heads(self):
        """Build separate prediction heads for each primitive parameter."""
        
        # Scale prediction: μ_x, μ_y, μ_z, σ_x, σ_y, σ_z (6 values)
        self.scale_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 6)
        )
        
        # Rotation prediction: Quaternion with uncertainty (8 values)
        # μ_w, μ_x, μ_y, μ_z, σ_w, σ_x, σ_y, σ_z
        self.rotation_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 8)
        )
        
        # Translation prediction: μ_x, μ_y, μ_z, σ_x, σ_y, σ_z (6 values)
        self.translation_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 6)
        )
        
        # Class prediction: logits for n_classes
        self.class_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.n_classes)
        )
        
        # EOS prediction: binary logit for end-of-sequence
        self.eos_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, 1)
        )
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        point_features: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            point_features: (B, N_points, D_feature) - per-point features
            point_mask: (B, N_points) - optional mask (True = valid)
            
        Returns:
            scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
            rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
            translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
            class_logits: (B, N_primitives, n_classes) - class logits
            eos_logits: (B, N_primitives, 1) - end-of-sequence logits
        """
        batch_size = point_features.shape[0]
        
        # Project point features
        point_features = self.point_feature_proj(point_features)
        
        # Create primitive queries
        query_indices = torch.arange(self.n_primitives, device=point_features.device)
        queries = self.primitive_queries(query_indices)
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Prepend SOS token
        sos = self.sos_token.expand(batch_size, -1, -1)
        queries = torch.cat([sos, queries], dim=1)
        
        # Add positional encoding
        queries = self.query_pos_encoding(queries)
        
        # Attention mask for point features
        memory_key_padding_mask = None
        if point_mask is not None:
            memory_key_padding_mask = ~point_mask
        
        # Transformer decoder
        decoded = self.transformer_decoder(
            tgt=queries,
            memory=point_features,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Remove SOS token
        primitive_features = decoded[:, 1:, :]
        
        # Apply prediction heads
        scale_params = self.scale_head(primitive_features)
        rotation_params = self.rotation_head(primitive_features)
        translation_params = self.translation_head(primitive_features)
        class_logits = self.class_head(primitive_features)
        eos_logits = self.eos_head(primitive_features)
        
        # Post-process to ensure positive σ
        scale_params = self._postprocess_params(scale_params)
        rotation_params = self._postprocess_params(rotation_params)
        translation_params = self._postprocess_params(translation_params)
        
        return scale_params, rotation_params, translation_params, class_logits, eos_logits
    
    def _postprocess_params(self, params: torch.Tensor) -> torch.Tensor:
        """Ensure σ values are positive using softplus."""
        dim = params.shape[-1]
        half = dim // 2
        mu = params[..., :half]
        sigma = F.softplus(params[..., half:]) + 1e-6  # Add small constant for numerical stability
        return torch.cat([mu, sigma], dim=-1)



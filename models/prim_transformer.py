import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from models.primitive_anything.michelangelo import ShapeConditioner as ShapeConditioner_miche
from pathlib import Path


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
        d_primitive_embedding: int = 16,
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

        self.michelangelo = ShapeConditioner_miche(dim_latent=256)

        for param in self.michelangelo.parameters():
            param.requires_grad = False
        self.michelangelo.eval()
        
        dim_model_out = self.michelangelo.dim_model_out
        self.to_cond_dim = nn.Linear(dim_model_out * 2, d_model)
        self.to_cond_dim_head = nn.Linear(dim_model_out, d_model)
        
        # Projection layer from input embedding to d_model
        self.input_d_model_proj = nn.Linear(10 + d_primitive_embedding, d_model)
        
        # Project point features to model dimension
        # self.point_feature_proj = nn.Linear(point_feature_dim, d_model)
        
        # Learnable query embeddings for primitives
        self.primitive_encoder = nn.Embedding(n_classes, d_primitive_embedding)
        
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
            nn.Linear(self.d_model, 6),
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
        sequence: Optional[torch.Tensor] = None,
        point_cloud: Optional[torch.Tensor] = None,
        point_mask: Optional[torch.Tensor] = None,
        point_features: Optional[torch.Tensor] = None,
        output_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            sequence: (B, T, 10 + n_classes + 1) - predicted sequence so far
            point_cloud: (B, N_points, 3) - raw point cloud
            point_features: (B, N_points, D) - pre-encoded features
            point_mask: (B, N_points) - optional mask (True = valid)
            
        Returns:
            scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
            rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
            translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
            class_logits: (B, N_primitives, n_classes) - class logits
            eos_logits: (B, N_primitives, 1) - end-of-sequence logits
        """
        assert point_cloud is not None or point_features is not None

        # Use SOS token if 
        
        # Extract point features
        if point_features is None:
            with torch.no_grad():
                pc_head, pc_embed = self.michelangelo(shape=point_cloud)
            # Project features (trainable)
            point_features = torch.cat([
                self.to_cond_dim_head(pc_head),      # (B, 1, d_model)
                self.to_cond_dim(pc_embed)            # (B, seq_len, d_model)
            ], dim=-2)
        batch_size = point_features.shape[0]
        
        # Project point features
        # point_features = self.point_feature_proj(point_features)
        
        # Prepare SOS token
        sos = self.sos_token.expand(batch_size, -1, -1) # (B, 1, d_model)

        if sequence is not None:
            # Encode primitive classes
            primitive_indices = sequence[:, :, -1].int()
            primitive_embeddings = self.primitive_encoder(primitive_indices) # (B, T, D)
            # Replace primitive classes with primitive embeddings
            sequence = torch.concat(
                [sequence[:, :, :10], primitive_embeddings], dim=-1
            ) # (B, T, 10 + D)
            # Project the input to d_model
            sequence = self.input_d_model_proj(sequence) # (B, 1 + T, d_model)
            # Preprend SOS token
            sequence = torch.concat([sos, sequence], dim=1) # (B, 1 + T, d_model)
        else:
            sequence = sos
        
        # Add positional encoding
        # Santiago: Removed this because order does not matter in our setting
        # queries = self.query_pos_encoding(sequence)
        
        # Attention mask for point features
        memory_key_padding_mask = None
        if point_mask is not None:
            memory_key_padding_mask = ~point_mask
        
        # Transformer decoder
        decoded = self.transformer_decoder(
            tgt=sequence,
            memory=point_features,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        
        # Get the last predicted timestep
        primitive_features = decoded[:, -1:, :]
        
        # Apply prediction heads
        scale_params = self.scale_head(primitive_features)
        rotation_params = self.rotation_head(primitive_features)
        translation_params = self.translation_head(primitive_features)
        class_logits = self.class_head(primitive_features)
        eos_logits = self.eos_head(primitive_features)
        
        # Post-process to ensure positive scale values
        scale_params = torch.concat(
            [torch.nn.functional.softplus(scale_params[:, :, :3]), scale_params[:, :, 3:]],
            dim=-1
        )
        # scale_params = self._postprocess_params(scale_params)
        # rotation_params = self._postprocess_params(rotation_params)
        # translation_params = self._postprocess_params(translation_params)
        
        return scale_params, rotation_params, translation_params, class_logits, eos_logits, point_features
    
    # def _postprocess_params(self, params: torch.Tensor) -> torch.Tensor:
    #     """Ensures values are positive using softplus."""
    #     dim = params.shape[-1]
    #     half = dim // 2
    #     mu = params[..., :half]
    #     sigma = params[..., half:]  # Add small constant for numerical stability
    #     return torch.cat([mu, sigma], dim=-1)
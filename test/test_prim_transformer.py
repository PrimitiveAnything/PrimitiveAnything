from models.prim_transformer import PrimitiveTransformerQuaternion
import torch

def test_primitive_transformer():
   print("Testing PrimitiveTransformerQuaternion...")
   
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # Test without Michelangelo (with pre-encoded features)
   print("\n=== Test 1: Without Michelangelo (pre-encoded features) ===")

   model = PrimitiveTransformerQuaternion(
      n_primitives=10,
      d_model=256,
      n_heads=8,
      n_layers=6,
      n_classes=3
   )
   model.to(device)
   
   # Create dummy input (pre-encoded features)
   batch_size = 2
   n_points = 2048
   point_features = torch.randn(batch_size, n_points, 256, device=device)
   
   # Forward pass
   scale_params, rotation_params, translation_params, class_logits, eos_logits, point_features = model(point_features=point_features)
   
   # Check shapes
   assert scale_params.shape == (2, 1, 6), f"Expected (2, 1, 6), got {scale_params.shape}"
   assert rotation_params.shape == (2, 1, 8), f"Expected (2, 1, 8), got {rotation_params.shape}"
   assert translation_params.shape == (2, 1, 6), f"Expected (2, 1, 6), got {translation_params.shape}"
   assert class_logits.shape == (2, 1, 3), f"Expected (2, 1, 3), got {class_logits.shape}"
   assert eos_logits.shape == (2, 1, 1), f"Expected (2, 1, 1), got {eos_logits.shape}"
   
   print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
   
   # Test with Michelangelo (with raw point cloud)
   print("\n=== Test 2: With Michelangelo (raw point cloud) ===")
        
   # Create dummy input (raw point cloud)
   positions = torch.randn(batch_size, n_points, 3)
   normals = torch.randn(batch_size, n_points, 3)
   normals = torch.nn.functional.normalize(normals, p=2, dim=-1)  # Normalize normals
   point_cloud = torch.cat([positions, normals], dim=-1).to(device)
   
   # Forward pass
   scale_params, rotation_params, translation_params, class_logits, eos_logits, point_features = model(
      point_cloud=point_cloud
   )
   
   assert scale_params.shape == (2, 1, 6), f"Expected (2, 1, 6), got {scale_params.shape}"
   assert rotation_params.shape == (2, 1, 8), f"Expected (2, 1, 8), got {rotation_params.shape}"
   assert translation_params.shape == (2, 1, 6), f"Expected (2, 1, 6), got {translation_params.shape}"
   assert class_logits.shape == (2, 1, 3), f"Expected (2, 1, 3), got {class_logits.shape}"
   assert eos_logits.shape == (2, 1, 1), f"Expected (2, 10, 1), got {eos_logits.shape}"
   
   print("\n✅ Test 2 passed!")
   print(f"Model with Michelangelo has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
   
   # Count trainable vs frozen parameters
   trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
   frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
   print(f"Trainable: {trainable/1e6:.2f}M, Frozen (Michelangelo): {frozen/1e6:.2f}M")
    
   print("\n✅ All tests passed!")

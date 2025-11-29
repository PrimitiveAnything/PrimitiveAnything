"""
Test script to verify prim_transformer works correctly with ShapeNet dataloader
"""

import sys
import os
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, '/ocean/projects/cis250266p/kanand/l3d_project')

from dataloaders.shapenet import ShapeNetDataset
from models.prim_transformer import PrimitiveTransformerQuaternion, QuaternionUtils

def test_dataloader():
    """Test that dataloader produces correct format"""
    print("=" * 60)
    print("TEST 1: Dataloader Format")
    print("=" * 60)
    
    try:
        # Create dataset
        dataset = ShapeNetDataset(
            shapenet_dir="./data/shapenet/",  # Adjust path as needed
            n_sample_points=10000,
            normalize=True
        )
        print(f"✓ Dataset created with {len(dataset)} samples")
        
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0  # Set to 0 for debugging
        )
        print(f"✓ Dataloader created")
        
        # Get one batch
        batch = next(iter(dataloader))
        
        # Check shape
        print(f"\n✓ Batch shape: {batch.shape}")
        assert batch.shape[0] == 2, f"Expected batch size 2, got {batch.shape[0]}"
        assert batch.shape[1] == 10000, f"Expected 10000 points, got {batch.shape[1]}"
        assert batch.shape[2] == 6, f"Expected 6 channels, got {batch.shape[2]}"
        print(f"  Expected: (2, 10000, 6) ✓")
        
        # Check positions (first 3 channels)
        positions = batch[:, :, :3]
        pos_min = positions.min().item()
        pos_max = positions.max().item()
        print(f"\n✓ Position range: [{pos_min:.3f}, {pos_max:.3f}]")
        if abs(pos_min) > 1.0 or abs(pos_max) > 1.0:
            print(f"  ⚠ Warning: Positions outside [-1, 1]. Should be normalized!")
        else:
            print(f"  Good: Positions are normalized ✓")
        
        # Check normals (last 3 channels)
        normals = batch[:, :, 3:]
        normal_magnitudes = torch.norm(normals, p=2, dim=-1)
        mean_magnitude = normal_magnitudes.mean().item()
        print(f"\n✓ Normal magnitude (mean): {mean_magnitude:.3f}")
        if abs(mean_magnitude - 1.0) > 0.1:
            print(f"  ⚠ Warning: Normals should have magnitude ~1.0")
        else:
            print(f"  Good: Normals are unit vectors ✓")
        
        # Check for NaNs
        has_nan = torch.isnan(batch).any().item()
        print(f"\n✓ Contains NaN: {has_nan}")
        assert not has_nan, "Batch contains NaN values!"
        
        print("\n" + "=" * 60)
        print("✅ Dataloader test PASSED!")
        print("=" * 60)
        
        return batch
        
    except Exception as e:
        print(f"\n❌ Dataloader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_initialization():
    """Test that model initializes correctly"""
    print("\n" + "=" * 60)
    print("TEST 2: Model Initialization")
    print("=" * 60)
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Initialize model
        model = PrimitiveTransformerQuaternion(
            n_primitives=10,
            d_model=256,
            n_heads=8,
            n_layers=6,
            n_classes=3,
            use_michelangelo=True,
        )
        print("✓ Model initialized")
        
        # Move to device
        model = model.to(device)
        model.eval()
        print(f"✓ Model moved to {device}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n✓ Total parameters: {total_params/1e6:.2f}M")
        print(f"  Trainable: {trainable_params/1e6:.2f}M")
        print(f"  Frozen (Michelangelo): {frozen_params/1e6:.2f}M")
        
        print("\n" + "=" * 60)
        print("✅ Model initialization test PASSED!")
        print("=" * 60)
        
        return model, device
        
    except Exception as e:
        print(f"\n❌ Model initialization FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_forward_pass(model, device, batch):
    """Test forward pass with real data"""
    print("\n" + "=" * 60)
    print("TEST 3: Forward Pass")
    print("=" * 60)
    
    if model is None or batch is None:
        print("❌ Skipping forward pass (prerequisites failed)")
        return
    
    try:
        # Move batch to device
        batch = batch.to(device)
        print(f"✓ Batch moved to {device}")
        print(f"  Batch shape: {batch.shape}")
        
        # Forward pass
        print("\nRunning forward pass...")
        with torch.no_grad():
            scale_params, rotation_params, translation_params, class_logits, eos_logits = model(batch)
        
        print("✓ Forward pass successful!")
        
        # Check output shapes
        batch_size = batch.shape[0]
        n_prims = 10
        
        print(f"\n✓ Output shapes:")
        print(f"  scale_params: {scale_params.shape} (expected: ({batch_size}, {n_prims}, 6))")
        assert scale_params.shape == (batch_size, n_prims, 6), f"Wrong shape for scale_params"
        
        print(f"  rotation_params: {rotation_params.shape} (expected: ({batch_size}, {n_prims}, 8))")
        assert rotation_params.shape == (batch_size, n_prims, 8), f"Wrong shape for rotation_params"
        
        print(f"  translation_params: {translation_params.shape} (expected: ({batch_size}, {n_prims}, 6))")
        assert translation_params.shape == (batch_size, n_prims, 6), f"Wrong shape for translation_params"
        
        print(f"  class_logits: {class_logits.shape} (expected: ({batch_size}, {n_prims}, 3))")
        assert class_logits.shape == (batch_size, n_prims, 3), f"Wrong shape for class_logits"
        
        print(f"  eos_logits: {eos_logits.shape} (expected: ({batch_size}, {n_prims}, 1))")
        assert eos_logits.shape == (batch_size, n_prims, 1), f"Wrong shape for eos_logits"
        
        # Check for NaNs in outputs
        has_nan = (
            torch.isnan(scale_params).any() or
            torch.isnan(rotation_params).any() or
            torch.isnan(translation_params).any() or
            torch.isnan(class_logits).any() or
            torch.isnan(eos_logits).any()
        )
        print(f"\n✓ Outputs contain NaN: {has_nan}")
        assert not has_nan, "Outputs contain NaN values!"
        
        # Check sigma values are positive
        scale_sigma = scale_params[..., 3:]
        rotation_sigma = rotation_params[..., 4:]
        translation_sigma = translation_params[..., 3:]
        
        print(f"\n✓ Sigma values (should all be > 0):")
        print(f"  scale_sigma: min={scale_sigma.min():.6f}, max={scale_sigma.max():.6f}")
        print(f"  rotation_sigma: min={rotation_sigma.min():.6f}, max={rotation_sigma.max():.6f}")
        print(f"  translation_sigma: min={translation_sigma.min():.6f}, max={translation_sigma.max():.6f}")
        
        assert (scale_sigma > 0).all(), "Scale sigma has non-positive values!"
        assert (rotation_sigma > 0).all(), "Rotation sigma has non-positive values!"
        assert (translation_sigma > 0).all(), "Translation sigma has non-positive values!"
        
        # Test quaternion operations
        print(f"\n✓ Testing quaternion operations:")
        quat_mu = rotation_params[..., :4]
        quat_normalized = QuaternionUtils.normalize_quaternion(quat_mu)
        quat_norms = torch.norm(quat_normalized, p=2, dim=-1)
        
        print(f"  Quaternion norms: min={quat_norms.min():.6f}, max={quat_norms.max():.6f}")
        assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5), \
            "Normalized quaternions should have unit norm!"
        
        # Convert to rotation matrices
        R = QuaternionUtils.quaternion_to_matrix(quat_normalized)
        print(f"  Rotation matrix shape: {R.shape} (expected: ({batch_size}, {n_prims}, 3, 3))")
        
        # Check orthonormality
        RTR = torch.matmul(R.transpose(-2, -1), R)
        I = torch.eye(3, device=R.device).unsqueeze(0).unsqueeze(0).expand_as(RTR)
        is_orthonormal = torch.allclose(RTR, I, atol=1e-4)
        print(f"  Rotation matrices orthonormal: {is_orthonormal}")
        assert is_orthonormal, "Rotation matrices are not orthonormal!"
        
        # Check determinant
        det = torch.det(R)
        det_correct = torch.allclose(det, torch.ones_like(det), atol=1e-4)
        print(f"  Determinants are +1: {det_correct}")
        assert det_correct, "Determinants should be +1!"
        
        print("\n" + "=" * 60)
        print("✅ Forward pass test PASSED!")
        print("=" * 60)
        
        return scale_params, rotation_params, translation_params, class_logits, eos_logits
        
    except Exception as e:
        print(f"\n❌ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_sampling(scale_params, rotation_params, translation_params, class_logits, eos_logits):
    """Test sampling from the predicted distributions"""
    print("\n" + "=" * 60)
    print("TEST 4: Sampling from Distributions")
    print("=" * 60)
    
    if scale_params is None:
        print("❌ Skipping sampling test (prerequisites failed)")
        return
    
    try:
        # Extract means and sigmas
        scale_mu = scale_params[..., :3]
        scale_sigma = scale_params[..., 3:]
        
        rotation_mu = rotation_params[..., :4]
        rotation_sigma = rotation_params[..., 4:]
        
        translation_mu = translation_params[..., :3]
        translation_sigma = translation_params[..., 3:]
        
        print("✓ Extracted means and sigmas")
        
        # Sample
        scale_sample = scale_mu + scale_sigma * torch.randn_like(scale_mu)
        rotation_sample = rotation_mu + rotation_sigma * torch.randn_like(rotation_mu)
        translation_sample = translation_mu + translation_sigma * torch.randn_like(translation_mu)
        
        print("✓ Sampled from distributions")
        
        # Normalize quaternion
        rotation_sample = QuaternionUtils.normalize_quaternion(rotation_sample)
        print("✓ Normalized quaternion samples")
        
        # Sample class
        class_probs = torch.softmax(class_logits, dim=-1)
        class_dist = torch.distributions.Categorical(class_probs)
        class_sample = class_dist.sample()
        
        print("✓ Sampled class labels")
        
        # EOS
        eos_prob = torch.sigmoid(eos_logits.squeeze(-1))
        
        print("✓ Computed EOS probabilities")
        
        # Assemble predParts (B, N, 12)
        predParts = torch.cat([
            scale_sample,  # (B, N, 3)
            rotation_sample,  # (B, N, 4)
            translation_sample,  # (B, N, 3)
            eos_prob.unsqueeze(-1),  # (B, N, 1)
            class_sample.unsqueeze(-1).float(),  # (B, N, 1)
        ], dim=-1)
        
        print(f"\n✓ Assembled predParts: {predParts.shape}")
        assert predParts.shape[-1] == 12, f"Expected 12 values per primitive, got {predParts.shape[-1]}"
        
        print(f"\n✓ Sample statistics:")
        print(f"  Scale range: [{scale_sample.min():.3f}, {scale_sample.max():.3f}]")
        print(f"  Translation range: [{translation_sample.min():.3f}, {translation_sample.max():.3f}]")
        print(f"  Class distribution: {class_sample.unique(return_counts=True)}")
        print(f"  EOS prob range: [{eos_prob.min():.3f}, {eos_prob.max():.3f}]")
        
        print("\n" + "=" * 60)
        print("✅ Sampling test PASSED!")
        print("=" * 60)
        
        return predParts
        
    except Exception as e:
        print(f"\n❌ Sampling test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "TESTING PRIM_TRANSFORMER WITH SHAPENET" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    # Test 1: Dataloader
    batch = test_dataloader()
    
    # Test 2: Model initialization
    model, device = test_model_initialization()
    
    # Test 3: Forward pass
    outputs = test_forward_pass(model, device, batch)
    
    # Test 4: Sampling
    if outputs is not None:
        scale_params, rotation_params, translation_params, class_logits, eos_logits = outputs
        predParts = test_sampling(scale_params, rotation_params, translation_params, class_logits, eos_logits)
    
    # Final summary
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 20 + "FINAL SUMMARY" + " " * 25 + "║")
    print("╚" + "=" * 58 + "╝")
    print("\n")
    
    if batch is not None and model is not None and outputs is not None:
        print("🎉 ALL TESTS PASSED! 🎉")
        print("\nYour setup is ready for training!")
        print("\nNext steps:")
        print("  1. Integrate this into your training loop")
        print("  2. Add loss computation (coverage + consistency)")
        print("  3. Add REINFORCE logic for stochastic actions")
        print("  4. Start training!")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding.")
    
    print("\n")


if __name__ == "__main__":
    main()
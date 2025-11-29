import torch
import pytest

from utils.get_primitives import get_samples, get_primitives
from primitives import (
    CuboidSurface,
    EllipsoidSurface,
    EllipticalCylinderSurface,
    EmptySurface
)


# --------------------------
# get_samples() TESTS
# --------------------------

def test_get_samples_shape():
    B = 4
    emb = torch.zeros((B, 1, 23))
    out, _ = get_samples(emb)
    assert out.shape == (B, 1, 11), "Output must be B x 1 x 11"


def test_get_samples_class_argmax():
    """
    Ensure the last dim index corresponds to argmax of logits.
    """
    B = 3
    emb = torch.zeros((B, 1, 23))

    # logits for 3 classes at positions 20:23
    emb[:, :, 20:] = torch.tensor([[-1.0, 5.0, 0.0]])  # class=1 is max

    out, _ = get_samples(emb)
    assert torch.all(out[:, :, 10] == 1), "Class index should be argmax(logits)"


def test_get_samples_sampling_effect():
    """
    Ensure sampling incorporates mean + std.
    """
    B = 2
    emb = torch.zeros((B, 1, 23))

    # scale mean = (1,1,1), std = (1,1,1)
    emb[:, :, 0:3] = 1.0
    emb[:, :, 3:6] = 1.0

    out1, _ = get_samples(emb)
    out2, _ = get_samples(emb)

    # random sampling should differ
    assert not torch.allclose(out1[:, :, :3], out2[:, :, :3]), \
        "Scale samples should differ due to sampling randomness."


# --------------------------
# get_primitives() TESTS
# --------------------------

def test_get_primitives_correct_classes():
    """
    Ensure the correct primitive class is chosen based on prim_type.
    """
    samples = torch.zeros((1, 3, 11))

    samples[0, 0, 10] = 0  # cuboid
    samples[0, 1, 10] = 1  # ellipsoid
    samples[0, 2, 10] = 2  # elliptical cylinder

    prims = get_primitives(samples)[0]

    assert isinstance(prims[0], CuboidSurface)
    assert isinstance(prims[1], EllipsoidSurface)
    assert isinstance(prims[2], EllipticalCylinderSurface)


def test_get_primitives_parameter_pass_through():
    """
    Ensure the scale, rotation, translation are passed correctly
    into the primitive constructors.
    """
    samples = torch.zeros((1, 1, 11))
    samples[0, 0, :3] = torch.tensor([2.0, 3.0, 4.0])  # scale
    samples[0, 0, 3:7] = torch.tensor([0.1, 0.2, 0.3, 0.4])  # rotation
    samples[0, 0, 7:10] = torch.tensor([9.0, 8.0, 7.0])  # translation
    samples[0, 0, 10] = 0  # CuboidSurface

    prim = get_primitives(samples)[0][0]

    assert torch.allclose(prim.scale, torch.tensor([2., 3., 4.]))
    assert torch.allclose(prim.quaternion, torch.tensor([0.1, 0.2, 0.3, 0.4]))
    assert torch.allclose(prim.translation, torch.tensor([9., 8., 7.]))


def test_get_primitives_batching():
    """
    Ensure batching returns the correct nested list structure.
    """
    samples = torch.zeros((2, 4, 11))  # 2 batches, 4 parts each
    prims = get_primitives(samples, 4)

    assert len(prims) == 2
    assert len(prims[0]) == 4
    assert len(prims[1]) == 4
    assert all(isinstance(p, CuboidSurface) for p in prims[0])


def test_get_primitives_padding_short_sequence():
    # Batch of 1, sequence length 2, max length 5
    B, T, max_len = 1, 2, 5
    samples = torch.zeros((B, T, 11))
    
    # Set primitive types: first Cuboid, second Ellipsoid
    samples[0, 0, 10] = 0
    samples[0, 1, 10] = 1

    primitives = get_primitives(samples, max_len)
    
    assert len(primitives) == B
    assert len(primitives[0]) == max_len  # padded to max_len
    # Check first two are actual surfaces, rest are EmptySurface
    assert isinstance(primitives[0][0], CuboidSurface)
    assert isinstance(primitives[0][1], EllipsoidSurface)
    for p in primitives[0][2:]:
        assert isinstance(p, EmptySurface)


def test_get_primitives_no_padding_needed():
    # Batch of 1, sequence length 3, max length 3
    B, T, max_len = 1, 3, 3
    samples = torch.zeros((B, T, 11))
    samples[0, :, 10] = torch.tensor([0, 1, 2])  # Cuboid, Ellipsoid, EllipticalCylinder

    primitives = get_primitives(samples, max_len)
    
    assert len(primitives[0]) == max_len
    # Should be no EmptySurface padding
    assert all(not isinstance(p, EmptySurface) for p in primitives[0])
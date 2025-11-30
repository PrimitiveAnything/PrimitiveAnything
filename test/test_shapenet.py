import os
import pytest 
import torch

from dataloaders.shapenet import ShapeNetDataset

@pytest.fixture
def dataset():
    print(os.getcwd())
    return ShapeNetDataset(shapenet_dir='/Users/avi/Documents/cmu/Learning for 3D Vision/l3d_project/data/shapenet', n_sample_points=10)

def test_dataset_length(dataset):
    # Ensure the dataset has a positive length
    assert len(dataset) > 0

def test_sample_shape(dataset):
    sample = dataset[0]
    # Check the sample has shape (N, 6)
    assert isinstance(sample, torch.Tensor)
    assert sample.shape[1] == 6
    assert sample.shape[0] == dataset.n_sample_points

def test_points_normals_split(dataset):
    sample = dataset[0]
    points = sample[:, :3]
    normals = sample[:, 3:]
    
    # Check shapes
    assert points.shape == (dataset.n_sample_points, 3)
    assert normals.shape == (dataset.n_sample_points, 3)

def test_normals_unit_length(dataset):
    sample = dataset[0]
    normals = sample[:, 3:]
    
    # Normals should be approximately unit vectors
    lengths = torch.norm(normals, dim=1)
    assert torch.allclose(lengths, torch.ones_like(lengths), atol=1e-5)

def test_multiple_samples_consistency(dataset):
    sample1 = dataset[0]
    sample2 = dataset[1]
    
    # Check that two samples are not identical
    assert not torch.allclose(sample1, sample2)
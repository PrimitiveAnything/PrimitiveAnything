import pytest

# Test that shapenet.py can be imported
def test_import_shapenet():
   try:
      import dataloaders.shapenet as shapenet
   except ImportError as e:
      pytest.fail(f"Importing shapenet.py failed with error: {e}")

# Test that R2N2ShapeNetDataset can be instantiated
def test_r2n2shapenetdataset_instantiation():
   from dataloaders.shapenet import R2N2ShapeNetDataset
   try:
      dataset = R2N2ShapeNetDataset('train', r2n2_shapenet_dir="data/r2n2_shapenet")
   except Exception as e:
      pytest.fail(f"Instantiating R2N2ShapeNetDataset failed with error: {e}")

# Test that __getitem__ works and returns expected types
def test_r2n2shapenetdataset_getitem():
   from dataloaders.shapenet import R2N2ShapeNetDataset
   dataset = R2N2ShapeNetDataset('train', r2n2_shapenet_dir="data/r2n2_shapenet")
   try:
      shapenet_model, r2n2_model = dataset[0]
      assert isinstance(shapenet_model, dict), "shapenet_model should be a dictionary"
      assert isinstance(r2n2_model, dict), "r2n2_model should be a dictionary"
   except Exception as e:
      pytest.fail(f"Calling __getitem__ on R2N2ShapeNetDataset failed with error: {e}")
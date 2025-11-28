import pytest

def test_tsdf_pred():
  import numpy as np
  pdb.set_trace()
  predParts = Variable(torch.FloatTensor([0.2, 0.2, 0.2,
                                          -0.2, -0.2, -0.2,
                                          0.5, np.sqrt(0.25), np.sqrt(0.25), np.sqrt(0.25)]).view(1,1,10))
  samplePoints = Variable(torch.FloatTensor([-0.4, -0.4, -0.4]).view(1,1,3))

  loss = tsdf_pred(samplePoints, predParts)

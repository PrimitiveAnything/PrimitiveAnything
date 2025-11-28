import pdb

import torch
import torch.nn as nn
from modules.quatUtils import  quat_conjugate, quat_rot_module
from modules.quatUtils import  get_random_quat

# class Rotate(nn.Module):
#   def __init__(self,nP):
#     super(Rotate, self).__init__()
#     self.quatRotModule = QuatRotModule()
#     self.nP = nP
#
#   def forward(self, points,quat):
#     ## points isn Batch_size x P x 3,  #Bx4 quat vectors
#     quat_rep  = quat.unsqueeze(1)
#     nP = points.size(1)
#     quat_rep  = quat_rep.repeat(1, nP, 1)
#
#     zero_points = 0*points[:,:,0].clone().view(-1, nP, 1)
#     quat_points = torch.cat([zero_points, points],dim=2)
#
#     rotated_points = self.quatRotModule(quat_points, quat_rep) # B x  P x 3
#     return rotated_points

# class Translate(nn.Module):
#   def __init__(self, nP):
#     super(Translate, self).__init__()
#     self.nP = nP
#
#   def forward(self, points, trans):
#     nP = points.size(1)
#     trans_rep = trans.unsqueeze(1)
#     trans_rep = trans_rep.repeat(1, nP, 1)
#
#     return points + trans_rep

# -- input is BXnPX3 points, BX3 translation vectors, BX4 quaternions
# -- output is BXnPX3 points
# -- performs p_out = R*(p_in - t)

# -- performs p_out = R*(p_in - t) ## used to while computing TSDF
def rigidTsdf(points, trans, quat):
  p1 = translate_module(points, -1*trans)
  p2 = rotate_module(p1, quat)
  # -- performs p_out = R'*p_in + t
  return p2


def rigidPointsTransform(points, trans, quat):
  quatConj = quat_conjugate(quat)
  p1 = rotate_module(points, quatConj)
  p2 = translate_module(p1, trans)
  return p2

## points is Batch_size x P x 3,  #Bx4 quat vectors
def rotate_module(points, quat):
  nP = points.size(1)
  quat_rep = quat.repeat(1, nP, 1)

  zero_points = 0 * points[:, :, 0].clone().view(-1, nP, 1)
  quat_points = torch.cat([zero_points, points], dim=2)

  rotated_points = quat_rot_module(quat_points, quat_rep)  # B x  P x 3
  return rotated_points

def translate_module(points, trans):
  nP = points.size(1)
  trans_rep = trans.repeat(1, nP, 1)

  return points + trans_rep



from pyquaternion import Quaternion
from torch.autograd import Variable


def test_trans():
  trans_module = translate_module
  N = 3
  P = 5
  points = Variable(torch.rand(N, P, 3))

  trans  = Variable(torch.rand(N, 3))

  points_t = trans_module(points, trans)
  point_t2 = trans_module(points_t, -trans)

  assert torch.mean(torch.abs(points- point_t2)) < 1E-4, 'error  in translate module'

def test_rot():
  N = 3
  P = 4
  rotModule = rotate_module
  conjModule = QuatConjugateModule()
  quat, q = get_random_quat()
  quat = quat.squeeze(1)
  quat = quat.repeat(N, 1)
  points = Variable(torch.rand(N, P, 3))
  rot_points = rotModule(points,quat)
  quat_conj = conjModule(quat)
  rot_points2 = rotModule(rot_points, quat_conj)
  assert torch.mean(torch.abs(rot_points - rot_points2)) < 1E-4, 'error in rotation module'

if __name__ == "__main__":
  test_trans()
  test_rot()

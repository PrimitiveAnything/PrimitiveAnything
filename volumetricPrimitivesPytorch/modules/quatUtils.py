import torch
import torch.nn as nn
from torch.autograd import Variable

inds = torch.LongTensor([0, -1, -2, -3, 1, 0, 3, -2, 2, -3, 0, 1, 3, 2, -1, 0]).view(4, 4)


# class HamiltonProduct(nn.Module):
#   def __init__(self):
#     super(HamiltonProduct, self).__init__()
#     self.inds = torch.LongTensor([0, -1, -2, -3,
#                                   1, 0, 3, -2,
#                                   2, -3, 0, 1,
#                                   3, 2, -1, 0]).view(4, 4)
#
#   def forward(self, q1, q2):
#     # q1*q2 Batch_size X Points x 4, Batch_size X Points x 4
#     # outputs Batch_Size x Points x 4
#     q_size = q1.size()
#     q1 = q1.view(-1, 4)
#     q2 = q2.view(-1, 4)
#     q1_q2_prods = []
#     for i in range(4):
#       ## Hack to make 0 as positive sign. add 0.01 to all the values..
#       q2_permute_0 = q2[:, np.abs(self.inds[i][0])]
#       q2_permute_0 = q2_permute_0 * np.sign(self.inds[i][0] + 0.01)
#
#       q2_permute_1 = q2[:, np.abs(self.inds[i][1])]
#       q2_permute_1 = q2_permute_1 * np.sign(self.inds[i][1] + 0.01)
#
#       q2_permute_2 = q2[:, np.abs(self.inds[i][2])]
#       q2_permute_2 = q2_permute_2 * np.sign(self.inds[i][2] + 0.01)
#
#       q2_permute_3 = q2[:, np.abs(self.inds[i][3])]
#       q2_permute_3 = q2_permute_3 * np.sign(self.inds[i][3] + 0.01)
#       q2_permute =  torch.stack([q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=1)
#
#       q1q2_v1 = torch.sum(q1 * q2_permute, dim=1)
#       q1_q2_prods.append(q1q2_v1)
#
#     q_ham = torch.cat(q1_q2_prods, dim=1)
#     q_ham = q_ham.view(q_size)
#     return q_ham
#
#
# class QuatConjugateModule(nn.Module):
#   def __init__(self):
#     super(QuatConjugateModule, self).__init__()
#
#   def forward(self, q):
#     q_size = q.size()
#     q = q.view(-1, 4)
#
#     q0 = q[:, 0]
#     q1 = -1 * q[:, 1]
#     q2 = -1 * q[:, 2]
#     q3 = -1 * q[:, 3]
#
#     q_conj = torch.stack([q0, q1, q2, q3], dim=1)
#     q_conj = q_conj.view(q_size)
#     return q_conj
#
# class QuatRotModule(nn.Module):
#   def __init__(self):
#     super(QuatRotModule, self).__init__()
#     self.quat_conjugate = QuatConjugateModule()
#     self.hamilton_product = HamiltonProduct()
#
#   def forward(self, points, quats):
#     # points Batch_size x P x 4 , quats Batch_size x P x 4
#     # input vectors with real dimension is zero.
#     # output with Batch_size x P x 3
#
#     quatConjugate = self.quat_conjugate(quats)
#     mult = self.hamilton_product(quats, points)
#     mult = self.hamilton_product(mult, quatConjugate)
#     return mult[:,:, 1:4]


def hamilton_product(q1, q2):
  q_size = q1.size()
  # q1 = q1.view(-1, 4)
  # q2 = q2.view(-1, 4)
  q1_q2_prods = []
  for i in range(4):
    ## Hack to make 0 as positive sign. add 0.01 to all the values..
    q2_permute_0 = q2[:, :, np.abs(inds[i][0])]
    q2_permute_0 = q2_permute_0 * np.sign(inds[i][0] + 0.01)

    q2_permute_1 = q2[:, :, np.abs(inds[i][1])]
    q2_permute_1 = q2_permute_1 * np.sign(inds[i][1] + 0.01)

    q2_permute_2 = q2[:, :, np.abs(inds[i][2])]
    q2_permute_2 = q2_permute_2 * np.sign(inds[i][2] + 0.01)

    q2_permute_3 = q2[:, :, np.abs(inds[i][3])]
    q2_permute_3 = q2_permute_3 * np.sign(inds[i][3] + 0.01)
    q2_permute = torch.stack([q2_permute_0, q2_permute_1, q2_permute_2, q2_permute_3], dim=2)

    q1q2_v1 = torch.sum(q1 * q2_permute, dim=2, keepdim=True)
    q1_q2_prods.append(q1q2_v1)

  q_ham = torch.cat(q1_q2_prods, dim=2)
  # q_ham = q_ham.view(q_size)
  return q_ham


def quat_conjugate(quat):
  # quat = quat.view(-1, 4)

  q0 = quat[:, :, 0]
  q1 = -1 * quat[:, :, 1]
  q2 = -1 * quat[:, :, 2]
  q3 = -1 * quat[:, :, 3]

  q_conj = torch.stack([q0, q1, q2, q3], dim=2)
  return q_conj


def quat_rot_module(points, quats):
  quatConjugate = quat_conjugate(quats)
  mult = hamilton_product(quats, points)
  mult = hamilton_product(mult, quatConjugate)
  return mult[:, :, 1:4]


from pyquaternion import Quaternion
import numpy as np
from torch.autograd import Variable
import pdb


def test_quat_conjugate():
  N = 3
  P = 1
  quat, q = get_random_quat()
  quat = quat.repeat(N, P, 1)

  conjugate_quat_module = quat_conjugate
  quat_conj = conjugate_quat_module(quat)
  dot_p = torch.mean(
    (quat[:, :, 0] * quat_conj[:, :, 0]) + torch.sum(quat[:, :, 1:4] * -1 * quat_conj[:, :, 1:4], dim=2))
  assert torch.abs(1 - dot_p).data[0] < 1E-4, 'Conjugate is incorrect'


def get_random_quat():
  q = Quaternion.random()
  q_n = np.array(q.elements, dtype=np.float32)
  quat = Variable(torch.from_numpy(q_n).float()).view(1, -1).view(1, 1, -1)
  return quat, q


def test_hamilton_product():
  conjugate_quat_module = quat_conjugate
  quat1, q1 = get_random_quat()
  quat1_c = conjugate_quat_module(quat1)
  quat1 = quat1.repeat(10, 3, 1)
  quat1_c = quat1_c.repeat(10, 3, 1)
  quat_product = hamilton_product(quat1, quat1_c)
  assert np.abs(1 - torch.mean(torch.sum(quat_product.view(-1, 4), 1)).data[0]) < 1E-4, 'Test1 error hamilton product'
  quat1, q1 = get_random_quat()
  quat2, q2 = get_random_quat()
  quat_product = hamilton_product(quat1, quat2).data.numpy().squeeze()

  q_product = np.array((q1 * q2).elements, dtype=np.float32)
  assert np.mean(np.abs(quat_product - q_product)) < 1E-4, 'Error in hamilton test 2'


def test_quat_rot_module():
  N = 400
  P = 4
  conjugate_quat_module = quat_conjugate
  quat_rotate_module = quat_rot_module

  quat1, q1 = get_random_quat()
  quat1_c = conjugate_quat_module(quat1)
  quat1 = quat1.repeat(N, P, 1)
  quat1_c = quat1_c.repeat(N, P, 1)
  points = np.random.rand(N, P, 3).astype(np.float32)
  zeros_points = Variable(torch.Tensor(N, P, 1).fill_(0))
  points_v = torch.cat([zeros_points, Variable(torch.from_numpy(points))], dim=2)
  rot_points = quat_rotate_module(points_v, quat1)
  rot_points2 = torch.cat([zeros_points, rot_points], dim=2)
  rot_points_back = quat_rotate_module(rot_points2, quat1_c)

  error = torch.mean(torch.abs(rot_points_back - points_v[:, :, 1:4])).data[0]
  # pdb.set_trace()
  assert error < 1E-4, 'error in quat rot {}'.format(error)
  assert True


if __name__ == "__main__":
  ## test all modules here
  torch.manual_seed(0)
  np.random.seed(0)

  test_quat_conjugate()
  test_hamilton_product()
  test_quat_rot_module()

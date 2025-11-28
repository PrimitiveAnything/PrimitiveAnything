import torch
import torch.nn as nn
from torch.autograd import Variable
# def sample_points_primitive(dimensions, count=150):
#
#   def sample_points_on_box(count_per_face):
#     # We will sample equally from all the 6 faces of the cube. This is not correct thing to do
#     points = torch.ones(count_per_face * 6, 3)
#
#     # x =1, y, and z drawn from
#     face_id = 0
#
#     points[face_id * count_per_face:(face_id + 1) * count_per_face] = sample_points_in_2d(count_per_face, '+yz')
#
#     # x =-1, y, and z drawn from
#     face_id = face_id + 1
#     points[face_id * count_per_face:(face_id + 1) * count_per_face] = sample_points_in_2d(count_per_face, '-yz')
#
#     # x , y = 1, and z drawn from
#     face_id = face_id + 1
#     points[face_id * count_per_face:(face_id + 1) * count_per_face] = sample_points_in_2d(count_per_face, 'x+z')
#
#     # x , y = -1, and z drawn from
#     face_id = face_id + 1
#     points[face_id * count_per_face:(face_id + 1) * count_per_face] = sample_points_in_2d(count_per_face, 'x-z')
#
#     # x , y, and z = 1 drawn from
#     face_id = face_id + 1
#     points[face_id * count_per_face:(face_id + 1) * count_per_face] = sample_points_in_2d(count_per_face, 'xy+')
#
#     # x , y, and z = -1 drawn from
#     face_id = face_id + 1
#     points[face_id * count_per_face:(face_id + 1) * count_per_face] = sample_points_in_2d(count_per_face, 'xy-')
#
#     return points
#
#   def sample_points_in_2d(count, face):
#     index = 0 if 'x' not in face else 1 if 'y' not in face else 2
#     value = -1 if '-' in face else 1
#     points = torch.Tensor(count, 3).uniform_(-1, 1)
#     points[:, index] = points[:, index] * 0 + value
#     return points
#
#   def get_area(dimensions, plane):
#     if 'y' not in plane:
#       return dimensions[0] * dimensions[2]
#     if 'x' not in plane:
#       return dimensions[1] * dimensions[2]
#     if 'z' not in plane:
#       return dimensions[0] * dimensions[1]
#
#   def importance_weights(dimensions, count_per_face):
#     weights = []
#     # x =1, y, and z drawn from
#     weights.append(get_area(dimensions, '+yz').expand(torch.Size([count_per_face, 1])))
#
#     # x =-1, y, and z drawn from
#     weights.append(get_area(dimensions, '-yz').expand(torch.Size([count_per_face, 1])))
#
#     # x , y = 1, and z drawn from
#     weights.append(get_area(dimensions, 'x+z').expand(torch.Size([count_per_face, 1])))
#
#     # x , y = -1, and z drawn from
#     weights.append(get_area(dimensions, 'x-z').expand(torch.Size([count_per_face, 1])))
#
#     # x , y, and z = 1 drawn from
#     weights.append(get_area(dimensions, 'xy+').expand(torch.Size([count_per_face, 1])))
#
#     # x , y, and z = -1 drawn from
#     weights.append(get_area(dimensions, 'xy-').expand(torch.Size([count_per_face, 1])))
#
#     weights = torch.cat(weights, dim=0)
#     weights = weights / ((1E-10 + torch.sum(weights)).expand(weights.size()))
#     return weights
#
#   def primitive_area(primitive):
#     return 2 * (primitive[0, 3] * (primitive[0, 4] + primitive[0, 5]) + primitive[0, 4] * primitive[0, 5])
#
#   count_per_face = count // 6
#   points = sample_points_on_box(count_per_face).cuda()
#   imp_weights = importance_weights(dimensions, count_per_face)
#   points_v = Variable(points)
#   return points_v, imp_weights, primitive_area(primitive)




# def sample_points(primitives):
#   primitive_sample_tuples = []
#   for primitive in primitives:
#     primitive_sample_tuples.append(sample_points_primitive(primitive, 150))
#
#   total_area = 0
#   for _, _, area in primitive_sample_tuples:
#     total_area = total_area + area.data[0]
#
#   total_area = 1
#   primitive_sample_tuples = [(k[0], k[1] / (total_area), k[2]) for k in primitive_sample_tuples]
#   points, weights, _ = zip(*primitive_sample_tuples)
#
#   points = torch.cat(points, dim=0)
#   weights = torch.cat(weights, dim=0)
#   if weights.sum().data[0] < 0:
#     pdb.set_trace()
#   return points, weights



class CuboidSurface(nn.Module):
  def __init__(self, nSamples, normFactor='None'):
    self.nSamples = nSamples
    self.samplesPerFace = nSamples // 3
    self.normFactor = normFactor


  def sample_wt_module(self, dims):
    # dims is bs x 1 x 3
    area = self.cuboidAreaModule(dims) # bs x 1 x 1
    dimsInv = dims.pow(-1)
    dimsInvNorm = dimsInv.sum(2, keepdim=True).repeat(1, 1, 3)
    normWeights = 3 * (dimsInv / dimsInvNorm)

    widthWt, heightWt, depthWt = torch.chunk(normWeights, chunks=3, dim=2)
    widthWt = widthWt.repeat(1, self.samplesPerFace, 1)
    heightWt = heightWt.repeat(1, self.samplesPerFace, 1)
    depthWt = depthWt.repeat(1, self.samplesPerFace, 1)

    sampleWt = torch.cat([widthWt, heightWt, depthWt], dim=1)
    finalWt = (1/self.samplesPerFace) * (sampleWt * area)
    return finalWt


  def sample(self,dims, coeff):
    dims_rep = dims.repeat(1,self.nSamples, 1)
    return dims_rep * coeff

  def cuboidAreaModule(self, dims):
    width, height, depth = torch.chunk(dims, chunks=3, dim=2)

    wh = width * height
    hd = height * depth
    wd = width * depth

    surfArea = 2*(wh + hd + wd)
    areaRep = surfArea.repeat(1, self.nSamples, 1)
    return areaRep

  def sample_points_cuboid(self, primShapes):
    # primPred B x 1 x 3
    # output B x nSamples x 3, B x nSamples x 1
    bs = primShapes.size(0)
    ns = self.nSamples
    nsp = self.samplesPerFace

    data_type = primShapes.data.type()
    coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nsp, 3).fill_(0.5)).type(data_type)
    coeffBernoulli = 2 * coeffBernoulli - 1   # makes entries -1 and 1

    coeff_w = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
    coeff_w[:, :, 0].copy_(coeffBernoulli[:,:,0].clone())

    coeff_h = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
    coeff_h[:, :, 1].copy_(coeffBernoulli[:,:,1].clone())

    coeff_d = torch.Tensor(bs, nsp, 3).type(data_type).uniform_(-1, 1)
    coeff_d[:, :, 2].copy_(coeffBernoulli[:,:,2].clone())

    coeff = torch.cat([coeff_w, coeff_h, coeff_d], dim=1)
    coeff = Variable(coeff)
    samples = self.sample(primShapes, coeff)
    importance_weights = self.sample_wt_module(primShapes)
    return samples, importance_weights



import pdb
def test_cuboid_surface():
  N = 1
  P = 1

  cuboidSampler = CuboidSurface(18)
  primShapes  = torch.Tensor(N, P, 3).fill_(0.5)

  samples, importance_weights = cuboidSampler.sample_points_cuboid(primShapes)

  pdb.set_trace()


if __name__ == "__main__":
  test_cuboid_surface()

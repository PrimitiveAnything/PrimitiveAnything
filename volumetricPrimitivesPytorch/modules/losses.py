import sys
sys.path.insert(0, '/home/nileshk/Research2/volumetricPrimitivesPytorch/')
from torch.autograd import Variable
from modules.transformer import rigidTsdf, rigidPointsTransform
from modules.quatUtils import quat_conjugate
from torch.nn import functional as F
import pdb
import torch


def cuboid_tsdf(sample_points, shape):
  ## sample_points Batch_size x nP x 3 , shape Batch_size x 1 x 3,
  ## output Batch_size x nP x 3
  nP = sample_points.size(1)
  shape_rep = shape.repeat(1, nP, 1)
  tsdf = torch.abs(sample_points) - shape_rep
  tsdfSq = F.relu(tsdf).pow(2).sum(dim=2)
  return tsdfSq  ## Batch_size x nP x 1


def tsdf_transform(sample_points, part):
  ## sample_points Batch_size x nP x 2, # parts Batch_size x 1 x 10
  shape = part[:, :, 0:3]  # B x 1 x 3
  trans = part[:, :, 3:6]  # B  x 1 x 3
  quat = part[:, :, 6:10]  # B x 1 x 4

  p1 = rigidTsdf(sample_points, trans, quat)  # B x nP x 3
  tsdf = cuboid_tsdf(p1, shape)  # B x nP x 1
  return tsdf


def get_existence_weights(tsdf, part):
  e = part[:,:,11:12]
  e = e.expand(tsdf.size())
  e = (1-e)*10
  return e


def tsdf_pred(sampledPoints, predParts):  ## coverage loss
  # sampledPoints  B x nP x 3
  # predParts  B x nParts x 10
  nParts = predParts.size(1)
  predParts = torch.chunk(predParts, nParts, dim=1)
  tsdfParts = []
  existence_weights = []
  for i in range(nParts):
    tsdf = tsdf_transform(sampledPoints, predParts[i])  # B x nP x 1
    tsdfParts.append(tsdf)
    existence_weights.append(get_existence_weights(tsdf, predParts[i]))


  existence_all = torch.cat(existence_weights, dim=2)
  tsdf_all = torch.cat(tsdfParts, dim=2) + existence_all
  tsdf_final = -1 * F.max_pool1d(-1 * tsdf_all, kernel_size=nParts)  # B x nP
  return tsdf_final


def primtive_surface_samples(predPart, cuboid_sampler):
  # B x 1 x 10
  shape = predPart[:, :, 0:3]  # B  x 1 x 3
  probs = predPart[:,:,11:12] # B x 1 x 1
  samples, imp_weights = cuboid_sampler.sample_points_cuboid(shape)
  probs = probs.expand(imp_weights.size())
  imp_weights = imp_weights * probs
  return samples, imp_weights


def partComposition(predParts, cuboid_sampler):
  # B x nParts x 10
  nParts = predParts.size(1)
  all_sampled_points = []
  all_sampled_weights = []
  predParts = torch.chunk(predParts, nParts, 1)
  for i in range(nParts):
    sampled_points, imp_weights = primtive_surface_samples(predParts[i], cuboid_sampler)
    transformedSamples = transform_samples(sampled_points, predParts[i])  # B x nPs x 3
    all_sampled_points.append(transformedSamples)  # B x nPs x 3
    all_sampled_weights.append(imp_weights)

  pointsOut = torch.cat(all_sampled_points, dim=1)  # b x nPs*nParts x 3
  weightsOut = torch.cat(all_sampled_weights, dim=1)  # b x nPs*nParts x 1
  return pointsOut, weightsOut


def transform_samples(samples, predParts):
  # B x nSamples x 3  , predParts B x 1 x 10
  trans = predParts[:, :, 3:6]  # B  x 1 x 3
  quat = predParts[:, :, 6:10]  # B x 1 x 4
  transformedSamples = rigidPointsTransform(samples, trans, quat)
  return transformedSamples


def normalize_weights(imp_weights):
  # B x nP x 1
  totWeights = (torch.sum(imp_weights, dim=1) + 1E-6).repeat(1, imp_weights.size(1), 1)
  norm_weights = imp_weights / totWeights
  return norm_weights


def chamfer_loss(predParts, dataloader, cuboid_sampler):
  sampled_points, imp_weights = partComposition(predParts, cuboid_sampler)
  norm_weights = normalize_weights(imp_weights)
  tsdfLosses = dataloader.chamfer_forward(sampled_points)
  weighted_loss = tsdfLosses * norm_weights  # B x nP x 1
  return torch.sum(weighted_loss, 1)



def test_tsdf_pred():
  import numpy as np
  pdb.set_trace()
  predParts = Variable(torch.FloatTensor([0.2, 0.2, 0.2,
                                          -0.2, -0.2, -0.2,
                                          0.5, np.sqrt(0.25), np.sqrt(0.25), np.sqrt(0.25)]).view(1,1,10))
  samplePoints = Variable(torch.FloatTensor([-0.4, -0.4, -0.4]).view(1,1,3))

  loss = tsdf_pred(samplePoints, predParts)

if __name__=="__main__":
  test_tsdf_pred()
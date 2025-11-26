import torch
import torch.nn as nn
import modules.netUtils as nUtils
import pdb
from torch.autograd import Variable

def meshGrid(minVal, maxVal, gridSize):
  # Xs, Ys, Zs = MeshGrid
  pointsX = torch.linspace(minVal[0], maxVal[0], gridSize[0])
  pointsY = torch.linspace(minVal[1], maxVal[1], gridSize[1])
  pointsZ = torch.linspace(minVal[2], maxVal[2], gridSize[2])
  # xs = torch.repeat(pointsX.view(-1, 1, 1, 1), 1, gridSize[1], gridSize[2], 1)
  # ys = torch.repeat(pointsY.view(1, -1, 1, 1), gridSize[0], 1, gridSize[2], 1)
  # zs = torch.repeat(pointsZ.view(1, 1, -1, 1), gridSize[0], gridSize[1], 1, 1)

  xs = pointsX.view(-1, 1, 1, 1).repeat(1, gridSize[1], gridSize[2], 1)
  ys = pointsY.view(1, -1, 1, 1).repeat(gridSize[0], 1, gridSize[2], 1)
  zs = pointsZ.view(1, 1, -1, 1).repeat(gridSize[0], gridSize[1], 1, 1)
  return torch.cat([xs, ys, zs],dim=3)



class ShapePredModule(nn.Module):
  def __init__(self, params):
    self.shapeModule = nn.Sequential()


class ShapePredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms=None):
    super(ShapePredModule, self).__init__()
    shapeLayer = nn.Conv3d(outChannelsV, params.nz, kernel_size=1)
    self.shapeLrDecay = params.shapeLrDecay
    shapeLayer.apply(nUtils.weightsInit)
    shapeLayer.note = 'shapePred'
    self.gridBound = params.gridBound
    biasTerms = biasTerms
    if biasTerms is not None:
      shapeLayer.bias.data = biasTerms.shape.clone()
    self.shapeLayer = shapeLayer
    self.sigmoid = nn.Sigmoid()

  def forward(self, feature):
    x = self.shapeLayer(feature)
    x = x * self.shapeLrDecay
    x = self.sigmoid(x)
    x = x * self.gridBound
    x = x.view(feature.size(0), -1)
    return x

class QuatPredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms=None):
    super(QuatPredModule, self).__init__()
    quatLayer = nn.Conv3d(outChannelsV, 4, kernel_size=1)
    self.shapeLrDecay = params.shapeLrDecay or 1
    quatLayer.apply(nUtils.weightsInit)
    quatLayer.note = 'quatLayer'

    biasTerms = biasTerms
    try:
      quatLayer.bias.data = biasTerms.quat.clone()
    except AttributeError:
      print("No bias init for qaut")
    self.quatLayer = quatLayer

  def forward(self, feature):
    x = self.quatLayer(feature)
    x = x.view(x.size(0), -1)
    x = normalize(x)
    return x


def normalize(x):
  x  = x/ (1E-12 + torch.norm(x, dim=1, p=2).expand(x.size()))
  return x

class TransPredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms=None):
    super(TransPredModule, self).__init__()
    transLayer = nn.Conv3d(outChannelsV, params.nz, kernel_size=1)
    # transLayer.apply(nUtils.weightsInit)
    transLayer.note = 'transPred'
    self.gridBound = params.gridBound
    biasTerms = biasTerms
    try:
      transLayer.bias.data = biasTerms.trans.clone()
    except AttributeError:
      assert True
      # print("No bias init for trans")
    self.transLayer = transLayer
    self.tanh = nn.Tanh()

  def forward(self, feature):
    x = self.transLayer(feature)
    x = self.tanh(x)
    x = x * self.gridBound
    x = x.view(x.size(0), -1)
    return x

class ProbPredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms=None):
    super(ProbPredModule, self).__init__()
    probLayer = nn.Conv3d(outChannelsV, 1, kernel_size=1)
    self.probLrDecay = params.probLrDecay
    probLayer.apply(nUtils.weightsInit)
    probLayer.note = 'probPred'
    self.gridBound = params.gridBound
    biasTerms = biasTerms
    if biasTerms is not None:
      probLayer.bias.data = biasTerms.prob.clone()
    self.probLayer = probLayer
    self.sigmoid = nn.Sigmoid()
    self.prune = params.prune

  def forward(self, feature):
    x = self.probLayer(feature)
    x = x * self.probLrDecay
    x = self.sigmoid(x)

    stocastic_outputs = x.view(feature.size(0), -1).bernoulli()
    # x = Variable(stocastic_outputs.data.clone())
    selections = stocastic_outputs
    if self.prune ==0:
      selections = Variable(torch.FloatTensor(x.size()).fill_(1).type_as(x.data))
    return torch.cat([x ,selections], dim=1), stocastic_outputs


class PrimitivePredModule(nn.Module):
  def __init__(self, params, outChannelsV, biasTerm):
    super(PrimitivePredModule, self).__init__()
    self.shapePred = ShapePredModule(params, outChannelsV, biasTerm)
    self.quatPred = QuatPredModule(params, outChannelsV, biasTerm)
    self.transPred = TransPredModule(params, outChannelsV, biasTerm)
    self.probPred = ProbPredModule(params, outChannelsV, biasTerm)

  def forward(self, feature):
    shape = self.shapePred(feature)
    quat = self.quatPred(feature)
    transPred = self.transPred(feature)
    probPred, stocastic_outputs = self.probPred(feature)
    output = torch.cat([shape, transPred, quat, probPred], dim = 1)
    return output, stocastic_outputs


class ReinforceShapeReward:
  def __init__(self, bMomentum, intrinsicReward, entropyWt=0):
    self.baseline = 0
    self.entropyWt = entropyWt
    self.bMomentum = bMomentum
    self.testMode = False
  def forward(self, rewards):
    self.baseline = self.bMomentum * self.baseline + (1 - self.bMomentum) * rewards.mean()
    rewards = rewards - self.baseline
    return rewards


class Primitives(nn.Module):
  def __init__(self, params, outChannelsV, biasTerms):
    super(Primitives, self).__init__()
    self.primitivePredModules = nn.ModuleList([PrimitivePredModule(params, outChannelsV, biasTerms)
                                               for i in range(params.nParts)])

  def forward(self, feature):
    primitives = []
    stocastic_actions = []
    for i, l in enumerate(self.primitivePredModules):
      output, stocastic_outputs = l(feature)
      primitives.append(output)
      stocastic_actions.append(stocastic_outputs)

    return torch.cat(primitives, dim=1), stocastic_actions  ## BatchSize x nParts*11
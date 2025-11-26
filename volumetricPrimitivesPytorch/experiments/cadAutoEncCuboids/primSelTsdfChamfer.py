
'''
CUDA_VISIBLE_DEVICES=1 python cadAutoEncCuboids/primSelTsdfChamfer.py
'''
import pdb
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../')))
import torch
import torch.nn as nn
import modules.volumeEncoder as vE
import modules.netUtils as netUtils
import modules.primitives as primitives

from torch.autograd import Variable
from data.cadConfigsChamfer import SimpleCadData
from modules.losses import tsdf_pred, chamfer_loss
from modules.cuboid import  CuboidSurface
import pdb
from modules.plotUtils import  plot3, plot_parts, plot_cuboid
import modules.marching_cubes as mc
import modules.meshUtils as mUtils
from modules.meshUtils import  savePredParts
from tensorboardX import SummaryWriter
torch.manual_seed(0)
from modules.config_utils import get_args


# params = lambda x: 0
#
# params.learningRate = 0.001
# params.meshSaveIter = 1000
# params.numTrainIter = 30000
# params.batchSize = 32
# params.batchSizeVis = 4
# params.visPower = 0.25
# params.lossPower = 2
# params.chamferLossWt = 1
# params.symLossWt = 1
# params.gridSize = 32
# params.gridBound = 0.5
# params.useBn = 1
# params.nParts = 20
# params.disp = 0
# params.imsave = 0
#
# params.gpu = 1
# params.visIter = 100
#
# params.prune = 1
# params.usePretrain = 1
#
# if params.usePretrain:
#   params.shapeLrDecay = 0.5
#   params.probLrDecay = 0.2
#   params.nullReward = 8e-5
# else:
#   params.shapeLrDecay = 0.01
#   params.probLrDecay = 0.0001
#   params.nullReward = 0
#
# # params.modelIter = 100000  # data loader reloads models after these many iterations
# params.modelIter = 2  # data loader reloads models after these many iterations
# #params.synset = 'chairs'  # chair:3001627, aero:2691156, table:4379243
# params.synset = '03001628'  # chair:3001627, aero:2691156, table:4379243
# params.name = 'chairChamferSurf_null_small_ft_prob0pt2_shape0pt5_null8em5_nan'
# params.bMomentum = 0.9  # baseline momentum for reinforce
# params.entropyWt = 0
#
# params.nSamplePoints = 1000
# params.nSamplesChamfer = 150  # number of points we'll sample per part
# params.useCubOnly = 0
#
# params.normFactor = 'Surf'
# params.pretrainNet = 'chairChamferSurf_null_small_init_prob0pt0001_shape0pt01_reinforce'
# params.pretrainLrShape = 0.01
# params.pretrainLrProb = 0.0001
# params.pretrainIter = 19000
#
# params.modelsDataDir = os.path.join('../cachedir/shapenet/chamferData/', params.synset)
# params.visDir = os.path.join('../cachedir/visualization/', params.name)
# params.visMeshesDir = os.path.join('../cachedir/visualization/meshes/', params.name)
# params.snapshotDir = os.path.join('../cachedir/snapshots/', params.name)

params = get_args()

params.modelsDataDir = os.path.join('../cachedir/shapenet/chamferData/', params.synset)
params.visDir = os.path.join('../cachedir/visualization/', params.name)
params.visMeshesDir = os.path.join('../cachedir/visualization/meshes/', params.name)
params.snapshotDir = os.path.join('../cachedir/snapshots/', params.name)

logger = SummaryWriter('../cachedir/logs/{}/'.format(params.name))

dataloader = SimpleCadData(params)
params.primTypes = ['Cu']
params.nz = 3
params.nPrimChoices = len(params.primTypes)
params.intrinsicReward = torch.Tensor(len(params.primTypes)).fill_(0)

if not os.path.exists(params.visDir):
  os.makedirs(params.visDir)

if not os.path.exists(params.visMeshesDir):
  os.makedirs(params.visMeshesDir)

if not os.path.exists(params.snapshotDir):
  os.makedirs(params.snapshotDir)

params.primTypesSurface = []
for p in range(len(params.primTypes)):
    params.primTypesSurface.append(params.primTypes[p])

part_probs = []

cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')
criterion  = nn.L1Loss()


def train(dataloader, netPred, reward_shaper, optimizer, iter):
  inputVol, tsdfGt, sampledPoints, loaded_cps = dataloader.forward()
  # pdb.set_trace()
  inputVol = Variable(inputVol.clone().cuda())
  tsdfGt = Variable(tsdfGt.cuda())
  sampledPoints = Variable(sampledPoints.cuda()) ## B x np x 3
  predParts, stocastic_actions = netPred.forward(inputVol) ## B x nPars*11
  predParts = predParts.view(predParts.size(0), params.nParts, 12)
  # pdb.set_trace()
  # pdb.set_trace()
  optimizer.zero_grad()
  tsdfPred= tsdf_pred(sampledPoints, predParts)
  # coverage = criterion(tsdfPred, tsdfGt)
  coverage_b = tsdfPred.mean(dim=1).squeeze()
  coverage = coverage_b.mean()
  consistency_b = chamfer_loss(predParts, dataloader, cuboid_sampler).squeeze()
  consistency = consistency_b.mean()
  loss = coverage_b + params.chamferLossWt*consistency_b
  rewards = []
  mean_reward = 0
  if params.prune ==1:
    reward = -1*loss.view(-1,1).data
    for i, action in enumerate(stocastic_actions):
      shaped_reward = reward - params.nullReward*torch.sum(action.data)
      shaped_reward = reward_shaper.forward(shaped_reward)
      action.reinforce(shaped_reward)
      rewards.append(shaped_reward)

    mean_reward = torch.stack(rewards).mean()

  logger.add_scalar('rewards/', mean_reward, iter)
  for i in range(params.nParts):
    logger.add_scalar('{}/prob'.format(i), predParts[:,i,10].data.mean(), iter)

  #
  # if iter % 200 == 0:
  #   # pdb.set_trace()
  #   # plot3(sampledPoints[0].data.cpu())
  #   for i in range(4):
  #     savePredParts(predParts[i], '../train_preds/train_{}.obj'.format(i))

  loss = torch.mean(loss)
  loss.backward()
  optimizer.step()

  return loss.data[0], coverage.data[0], consistency.data[0], mean_reward


import torch.nn as nn

class Network(nn.Module):
  def __init__(self, params):
    super(Network, self).__init__()
    self.ve = vE.convEncoderSimple3d(3,4,1,params.useBn)
    outChannels = self.outChannels = self.ve.output_channels
    layers = []
    for i in range(2):
      layers.append(nn.Conv3d(outChannels, outChannels,kernel_size=1))
      layers.append(nn.BatchNorm3d(outChannels))
      layers.append(nn.LeakyReLU(0.2,True))

    self.fc_layers = nn.Sequential(*layers)
    self.fc_layers.apply(netUtils.weightsInit)

    biasTerms = lambda x:0

    biasTerms.quat = torch.Tensor([1, 0, 0, 0])
    biasTerms.shape = torch.Tensor(params.nz).fill_(-3) / params.shapeLrDecay
    biasTerms.prob = torch.Tensor(1).fill_(0)
    for p in range(len(params.primTypes)):
      if (params.primTypes[p] == 'Cu'):
        biasTerms.prob[p] = 2.5 / params.probLrDecay

    self.primitivesTable = primitives.Primitives(params, outChannels, biasTerms)
    # self.primitivesTable.apply(netUtils.weightsInit)

  def forward(self, x):

    encoding  = self.ve(x)
    features = self.fc_layers(encoding)
    primitives, stocastic_actions = self.primitivesTable(features)
    return primitives, stocastic_actions

netPred = Network(params)
netPred.cuda()

reward_shaper = primitives.ReinforceShapeReward(params.bMomentum,  params.intrinsicReward, params.entropyWt)
# reward_shaper.cuda()

if params.usePretrain:
  updateShapeWtFunc = netUtils.scaleWeightsFunc(params.pretrainLrShape / params.shapeLrDecay, 'shapePred')
  updateProbWtFunc = netUtils.scaleWeightsFunc(params.pretrainLrProb / params.probLrDecay, 'probPred')
  updateBiasWtFunc = netUtils.scaleBiasWeights(params.probLrDecay, 'probPred')
  load_path = os.path.join('../cachedir/snapshots', params.pretrainNet, 'iter{}.pkl'.format(params.pretrainIter))
  netPretrain = torch.load(load_path)
  netPred.load_state_dict(netPretrain)
  print('Loading pretrained model from {}'.format(load_path))
  netPred.primitivesTable.apply(updateShapeWtFunc)
  netPred.primitivesTable.apply(updateProbWtFunc)
  # netPred.primitivesTable.apply(updateBiasWtFunc)



optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learningRate)

nSamplePointsTrain = params.nSamplePoints
nSamplePointsTest = params.gridSize**3

loss = 0
coverage = 0
consistency = 0
mean_reward = 0


def tsdfSqModTest(x):
  return torch.clamp(x,min=0).pow(2)



print("Iter\tErr\tTSDF\tChamf\tMeanRe")
for iter  in range(params.numTrainIter):
  print("{:10.7f}\t{:10.7f}\t{:10.7f}\t{:10.7f}\t{:10.7f}".format(iter, loss, coverage, consistency, mean_reward))
  loss, coverage, consistency, mean_reward = train(dataloader, netPred, reward_shaper, optimizer, iter)

  if iter % params.visIter ==0:
    reshapeSize = torch.Size([params.batchSizeVis, 1, params.gridSize, params.gridSize, params.gridSize])

    sample, tsdfGt, sampledPoints = dataloader.forwardTest()

    sampledPoints = sampledPoints[0:params.batchSizeVis].cuda()
    sample = sample[0:params.batchSizeVis].cuda()
    tsdfGt = tsdfGt[0:params.batchSizeVis].view(reshapeSize)

    tsdfGtSq = tsdfSqModTest(tsdfGt)
    netPred.eval()
    shapePredParams, _ = netPred.forward(Variable(sample))
    shapePredParams = shapePredParams.view(params.batchSizeVis, params.nParts, 12)
    netPred.train()

    if iter % params.meshSaveIter ==0:

      meshGridInit = primitives.meshGrid([-params.gridBound, -params.gridBound, -params.gridBound],
                                         [params.gridBound, params.gridBound, params.gridBound],
                                         [params.gridSize, params.gridSize, params.gridSize])
      predParams = shapePredParams
      for b in range(0, tsdfGt.size(0)):

        visTriSurf = mc.march(tsdfGt[b][0].cpu().numpy())
        mc.writeObj('{}/iter{}_inst{}_gt.obj'.format(params.visMeshesDir ,iter, b), visTriSurf)


        pred_b = []
        for px in range(params.nParts):
          pred_b.append(predParams[b,px,:].clone().data.cpu())

        mUtils.saveParts(pred_b, '{}/iter{}_inst{}_pred.obj'.format(params.visMeshesDir, iter, b))

  if ((iter+1) % 1000) == 0 :
    torch.save(netPred.state_dict() ,"{}/iter{}.pkl".format(params.snapshotDir,iter))

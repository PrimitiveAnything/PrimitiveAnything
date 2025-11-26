
'''
CUDA_VISIBLE_DEVICES=1 python cadAutoEncCuboids/primSelTsdfChamfer.py
'''
import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter

from dataloaders.cadConfigsChamfer import SimpleCadData


import modules.netUtils as netUtils
import modules.primitives as primitives
from modules.losses import tsdf_pred, chamfer_loss
from modules.cuboid import  CuboidSurface
from modules.plotUtils import  plot3, plot_parts, plot_cuboid
import modules.marching_cubes as mc
import modules.meshUtils as mUtils
from modules.meshUtils import  savePredParts
from modules.config_utils import get_args

from dataloaders.shapenet import R2N2ShapeNetDataset
from models.original import Network

torch.manual_seed(0)

params = get_args()

params.visDir = os.path.join('../cachedir/visualization/', params.name)
params.visMeshesDir = os.path.join('../cachedir/visualization/meshes/', params.name)
params.snapshotDir = os.path.join('../cachedir/snapshots/', params.name)

logger = SummaryWriter('../cachedir/logs/{}/'.format(params.name))

# Load dataset
train_dataset = R2N2ShapeNetDataset(partition='train', r2n2_shapenet_dir=params.modelsDataDir, synsets=params.synset, n_sample_points=params.nSamplePoints)
train_dataloader = DataLoader(train_dataset, batch_size=params.batchSize, shuffle=True, num_workers=4)

test_dataset = R2N2ShapeNetDataset(partition='test', r2n2_shapenet_dir=params.modelsDataDir, synsets=params.synset, n_sample_points=params.nSamplePoints)
test_dataloader = DataLoader(test_dataset, batch_size=params.batchSize, shuffle=False, num_workers=4)

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
  # Get batch
  for batch in dataloader:
    inputVol, tsdfGt, sampledPoints, loaded_cps = batch
    inputVol = Variable(inputVol.clone().cuda())
    tsdfGt = Variable(tsdfGt.cuda())
    sampledPoints = Variable(sampledPoints.cuda()) ## B x np x 3

    predParts, stocastic_actions = netPred.forward(inputVol) ## B x nPars*11
    predParts = predParts.view(predParts.size(0), params.nParts, 12)
    optimizer.zero_grad()
    tsdfPred= tsdf_pred(sampledPoints, predParts)
    # coverage = criterion(tsdfPred, tsdfGt)
    coverage_b = tsdfPred.mean(dim=1).squeeze()
    coverage = coverage_b.mean()
    consistency_b = chamfer_loss(predParts, loaded_cps, cuboid_sampler, params.gridBound, params.gridSize, inputVol).squeeze()
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

    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()

  return loss.data[0], coverage.data[0], consistency.data[0], mean_reward



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
  loss, coverage, consistency, mean_reward = train(test_dataloader, netPred, reward_shaper, optimizer, iter)

  if iter % params.visIter ==0:
    reshapeSize = torch.Size([params.batchSizeVis, 1, params.gridSize, params.gridSize, params.gridSize])

    for batch in test_dataloader:
      sample, tsdfGt, sampledPoints = batch

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

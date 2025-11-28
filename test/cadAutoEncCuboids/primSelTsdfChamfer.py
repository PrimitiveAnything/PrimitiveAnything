'''
python cadAutoEncCuboids/primSelTsdfChamfer.py
'''
import os
import sys
sys.path.insert(0, '/home/nileshk/Research2/volumetricPrimitivesPytorch/')
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
params = lambda x: 0

params.learningRate = 0.001
params.meshSaveIter = 10
params.numTrainIter = 20000
params.batchSize = 32
params.batchSizeVis = 4
# params.batchSizeTest = 1
params.visPower = 0.25
params.lossPower = 2
params.chamferLossWt = 1
params.symLossWt = 1
params.gridSize = 32
params.gridBound = 0.5
params.useBn = 1
params.nParts = 20
params.disp = 0
params.imsave = 0
params.shapeLrDecay = 0.01
params.probLrDecay = 0.0001
params.gpu = 1
params.visIter = 100
# params.modelIter = 100000  # data loader reloads models after these many iterations
params.modelIter = 2  # data loader reloads models after these many iterations
params.synset = 'chairs'  # chair:3001627, aero:2691156, table:4379243
# params.synset = '03001628'  # chair:3001627, aero:2691156, table:4379243
params.name = 'chairChamferSurf_null_small_init_prob0pt0001_shape0pt01'
params.bMomentum = 0.9  # baseline momentum for reinforce
params.entropyWt = 0
params.nullReward = 0.000
params.nSamplePoints = 1000
params.nSamplesChamfer = 150  # number of points we'll sample per part
params.useCubOnly = 0
params.usePretrain = 0
params.normFactor = 'Surf'
params.pretrainNet = 'chairChamferSurf_null_small_init_prob0pt0001_shape0pt01'
params.pretrainLrShape = 0.01
params.pretrainLrProb = 0.0001
params.pretrainIter = 20000
params.modelsDataDir = os.path.join('../cachedir/shapenet/chamferData/', params.synset)
params.visDir = os.path.join('../cachedir/visualization/', params.name)
params.visMeshesDir = os.path.join('../cachedir/visualization/meshes/', params.name)
params.infMeshDir = os.path.join('../cachedir/inference/meshes/', params.name)
params.snapshotDir = os.path.join('../cachedir/snapshots/', params.name)

dataloader = SimpleCadData(params)
params.nz = 3
params.primTypes = ['Cu']
params.nPrimChoices = len(params.primTypes)
params.intrinsicReward = torch.Tensor(len(params.primTypes)).fill_(0)



if not os.path.exists(params.infMeshDir):
  os.makedirs(params.infMeshDir)

params.primTypesSurface = []
for p in range(len(params.primTypes)):
    params.primTypesSurface.append(params.primTypes[p])


cuboid_sampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')
criterion  = nn.L1Loss()

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
    biasTerms.prob = torch.Tensor(len(params.primTypes)).fill_(0)
    for p in range(len(params.primTypes)):
      if (params.primTypes[p] == 'Cu'):
        biasTerms.prob[p] = 2.5 / params.probLrDecay

    self.primitivesTable = primitives.Primitives(params, outChannels, biasTerms)

  def forward(self, x):

    encoding  = self.ve(x)
    features = self.fc_layers(encoding)
    primitives = self.primitivesTable(features)
    return primitives

netPred = Network(params)
netPred.cuda()

optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learningRate)

nSamplePointsTrain = params.nSamplePoints
nSamplePointsTest = params.gridSize**3

loss = 0
coverage = 0
consitency = 0

import torch.nn.functional as F

def tsdfSqModTest(x):
  return torch.clamp(x,min=0).pow(2)




print("Iter\tErr\tTSDF\tChamf")
## Restore model from pre-trainnet.

preTrainNetDir = os.path.join('../cachedir/snapshots/', params.pretrainNet)
checkpoint_state_dict = torch.load("{}/iter{}.pkl".format(preTrainNetDir, params.pretrainIter))
netPred.load_state_dict(checkpoint_state_dict)
netPred.eval()
index = 0
while True:
    sample, tsdfGt, sampledPoints = dataloader.forwardTestSequential()
    if sample is None:
        break
    shapePredParams = netPred.forward(Variable(sample))
    shapePredParams = shapePredParams.view(params.batchSize, params.nParts, 10)
    predParams = shapePredParams
    for b in range(0, tsdfGt.size(0)):
        pred_b = []
        for px in range(params.nParts):
          pred_b.append(predParams[b,px,:].clone().data.cpu())
        mUtils.saveParts(pred_b, '{}/{}.obj'.format(params.infMeshDir,index))
        index += 1
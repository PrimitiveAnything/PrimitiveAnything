"""
CUDA_VISIBLE_DEVICES=1 python cadAutoEncCuboids/primSelTsdfChamfer.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from models.prim_transformer import PrimitiveTransformerQuaternion
from dataloaders.cadConfigsChamfer import SimpleCadData

from losses import coverage_loss, consistency_loss

import modules.netUtils as netUtils
import modules.primitives as primitives

from modules.plotUtils import plot3, plot_parts, plot_cuboid
import modules.marching_cubes as mc
import modules.meshUtils as mUtils
from modules.meshUtils import savePredParts
from modules.config_utils import get_args
from dataloaders.shapenet import R2N2ShapeNetDataset
from models.original import Network
from tqdm import tqdm

torch.manual_seed(0)


def train(dataloader, netPred, reward_shaper, optimizer, iter, params, logger, device):
    # Get batch
    netPred.train()
    progress_bar = tqdm(dataloader, desc="Epoch progress", leave=False)
    for batch in progress_bar:
        inputVol, sampledPoints = batch
        inputVol = inputVol.to(device)
        sampledPoints = sampledPoints.to(device)

        predParts, stocastic_actions = netPred.forward(inputVol.densities())  ## B x nPars*11
        predParts = predParts.view(predParts.size(0), params.nParts, 12)

        optimizer.zero_grad()
        cov_loss = coverage_loss(sampledPoints, predParts) # (B, N, 1)
        cons_loss = consistency_loss(predParts, params.nSamplesChamfer, sampledPoints, inputVol) # (B, N, 1)
        loss = cov_loss + params.chamferLossWt * cons_loss

        mean_reward = None
        if params.prune == 1:
            rewards = []
            reward = -1 * loss.view(-1, 1).data
            for i, action in enumerate(stocastic_actions):
                shaped_reward = reward - params.nullReward * torch.sum(action.data)
                shaped_reward = reward_shaper.forward(shaped_reward)
                action.reinforce(shaped_reward)
                rewards.append(shaped_reward)

            mean_reward = torch.stack(rewards).mean()

            logger.add_scalar("rewards/", mean_reward, iter)
            for i in range(params.nParts):
                logger.add_scalar(
                    "{}/prob".format(i), predParts[:, i, 10].data.mean(), iter
                )
        
        # Display metrics
        progress_bar.set_postfix_str(
            f"Total Loss: {loss.item():.4f}\tCoverage Loss: {cov_loss.item():.4f}\tConsistency Loss: {cons_loss.item():.4f}"
        )

        loss.backward()
        optimizer.step()

    return loss.item(), cov_loss.item(), cons_loss.item(), mean_reward


def main():

    params = get_args()
    params.visDir = os.path.join("output/visualization/", params.name)
    params.visMeshesDir = os.path.join("output/visualization/meshes/", params.name)
    params.snapshotDir = os.path.join("output/snapshots/", params.name)
    params.primTypes = ["Cu"]
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

    logger = SummaryWriter("logs/{}/".format(params.name))

    # Load dataset
    train_dataset = R2N2ShapeNetDataset(
        partition="train",
        r2n2_shapenet_dir=params.modelsDataDir,
        synsets=params.synset,
        n_sample_points=params.nSamplePoints,
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=params.batchSize, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn
    )
    test_dataset = R2N2ShapeNetDataset(
        partition="test",
        r2n2_shapenet_dir=params.modelsDataDir,
        synsets=params.synset,
        n_sample_points=params.nSamplePoints,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=params.batchSize, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn
    )

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model

    netPred = PrimitiveTransformerQuaternion(
        n_primitives=params.nParts,
        d_model=256,
        n_heads=8,
        n_layers=6,
        n_classes=len(params.primTypes)
    )
        
    if params.usePretrain:
        # Load model weights
        updateShapeWtFunc = netUtils.scaleWeightsFunc(
            params.pretrainLrShape / params.shapeLrDecay, "shapePred"
        )
        updateProbWtFunc = netUtils.scaleWeightsFunc(
            params.pretrainLrProb / params.probLrDecay, "probPred"
        )
        updateBiasWtFunc = netUtils.scaleBiasWeights(params.probLrDecay, "probPred")
        load_path = os.path.join(
            "../cachedir/snapshots",
            params.pretrainNet,
            "iter{}.pkl".format(params.pretrainIter),
        )
        netPretrain = torch.load(load_path)
        netPred.load_state_dict(netPretrain)
        print("Loading pretrained model from {}".format(load_path))
        netPred.primitivesTable.apply(updateShapeWtFunc)
        netPred.primitivesTable.apply(updateProbWtFunc)
        # netPred.primitivesTable.apply(updateBiasWtFunc)
    netPred.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learningRate)

    reward_shaper = primitives.ReinforceShapeReward(
        params.bMomentum, params.intrinsicReward, params.entropyWt
    )
    # reward_shaper.cuda()


    # Initialize training metrics

    # Train the model
    print("Iter\tErr\tTSDF\tChamf\tMeanRe")
    for iter in tqdm(range(params.numTrainIter), desc='Training progress'):
        loss, coverage, consistency, mean_reward = train(
            train_dataloader, netPred, reward_shaper, optimizer, iter, params, logger, device
        )
        print(
            "{:10.7f}\t{:10.7f}\t{:10.7f}\t{:10.7f}\t{:10.7f}".format(
                iter, loss, coverage, consistency, mean_reward
            )
        )

        # Visualize results
        if iter % params.visIter == 0 and False:
            reshapeSize = torch.Size(
                [
                    params.batchSizeVis,
                    1,
                    params.gridSize,
                    params.gridSize,
                    params.gridSize,
                ]
            )

            for batch in tqdm(test_dataloader, desc='Validation progress', leave=False):
                sample, tsdfGt, sampledPoints = batch

                sampledPoints = sampledPoints[0 : params.batchSizeVis].cuda()
                sample = sample[0 : params.batchSizeVis].cuda()
                tsdfGt = tsdfGt[0 : params.batchSizeVis].view(reshapeSize)

                netPred.eval()
                shapePredParams, _ = netPred.forward(Variable(sample))
                shapePredParams = shapePredParams.view(
                    params.batchSizeVis, params.nParts, 12
                )
                netPred.train()

                if iter % params.meshSaveIter == 0:

                    meshGridInit = primitives.meshGrid(
                        [-params.gridBound, -params.gridBound, -params.gridBound],
                        [params.gridBound, params.gridBound, params.gridBound],
                        [params.gridSize, params.gridSize, params.gridSize],
                    )
                    predParams = shapePredParams
                    for b in range(0, tsdfGt.size(0)):

                        visTriSurf = mc.march(tsdfGt[b][0].cpu().numpy())
                        mc.writeObj(
                            "{}/iter{}_inst{}_gt.obj".format(
                                params.visMeshesDir, iter, b
                            ),
                            visTriSurf,
                        )

                        pred_b = []
                        for px in range(params.nParts):
                            pred_b.append(predParams[b, px, :].clone().data.cpu())

                        mUtils.saveParts(
                            pred_b,
                            "{}/iter{}_inst{}_pred.obj".format(
                                params.visMeshesDir, iter, b
                            ),
                        )

        if ((iter + 1) % 1000) == 0:
            torch.save(
                netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter)
            )

if __name__ == '__main__':
    main()
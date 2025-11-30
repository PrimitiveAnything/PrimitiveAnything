"""
CUDA_VISIBLE_DEVICES=1 python cadAutoEncCuboids/primSelTsdfChamfer.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.autograd import Variable
from models.prim_transformer import PrimitiveTransformerQuaternion
from dataloaders.cadConfigsChamfer import SimpleCadData
from dataloaders.shapenet import ShapeNetDataset
from losses import coverage_loss, consistency_loss, chamfer_distance_loss
from pytorch3d.ops import sample_points_from_meshes

import modules.netUtils as netUtils
import modules.primitives as primitives

from modules.plotUtils import plot3, plot_parts, plot_cuboid
import modules.marching_cubes as mc
import modules.meshUtils as mUtils
from modules.meshUtils import savePredParts
from modules.config_utils import get_args
# from dataloaders.shapenet import R2N2ShSapeNetDataset
from models.original import Network
from tqdm import tqdm
from utils.get_primitives import get_samples, get_primitives
from primitives.compose import generate_mesh_from_primitives

torch.manual_seed(0)


def train(dataloader, netPred, optimizer, iter, params, device):
    # Get batch
    netPred.train()
    progress_bar = tqdm(dataloader, desc="Epoch progress", leave=False)
    for batch in progress_bar:
        sampledPoints = batch
        sampledPoints = sampledPoints.to(device)

        # scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
        # rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
        # translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
        # class_logits: (B, N_primitives, n_classes) - class logits
        # eos_logits: (B, N_primitives, 1) - end-of-sequence logits
        samples = []
        log_probs = []
        mask = torch.ones((sampledPoints.shape[0], netPred.n_primitives, 1), device=device)
        point_feats = None
        for t in range(netPred.n_primitives):
            scale, rot, transl, cls, eos, point_feats = netPred(sampledPoints, point_features=point_feats)

            embedding = torch.cat([scale, rot, transl, eos, cls], dim=-1)

            if not samples or (samples and samples[-1][:, t-1, 0] != 0):
                sample, log_prob = get_samples(embedding) # B x 1 x 11
            else:
                sample, log_prob = torch.zeros_like(samples[-1]), torch.zeros_like(log_probs[-1])
                
            samples.append(sample)
            log_probs.append(log_prob)

        samples = torch.concat(samples, dim=1)
        log_probs = torch.concat(log_probs, dim=1).sum(dim=1)
        primitives = get_primitives(samples, netPred.n_primitives)
        meshes = generate_mesh_from_primitives(primitives)
        predPoints = sample_points_from_meshes(meshes, 10000)

        # cov_loss = coverage_loss(sampledPoints, predParts) # (B, N, 1)
        # cons_loss = consistency_loss(predParts, params.nSamplesChamfer, sampledPoints, inputVol) # (B, N, 1)
        # loss = cov_loss + params.chamferLossWt * cons_loss
        loss, _ = chamfer_distance_loss(predPoints, sampledPoints, point_reduction='mean')

        # Display metrics
        progress_bar.set_postfix_str(
            f"Total Loss: {loss.item():.4f}"
        )

        loss *= log_probs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def main():

    params = get_args()
    params.visDir = os.path.join("output/visualization/", params.name)
    params.visMeshesDir = os.path.join("output/visualization/meshes/", params.name)
    params.snapshotDir = os.path.join("output/snapshots/", params.name)
    params.primTypes = 3

    if not os.path.exists(params.visDir):
        os.makedirs(params.visDir)

    if not os.path.exists(params.visMeshesDir):
        os.makedirs(params.visMeshesDir)

    if not os.path.exists(params.snapshotDir):
        os.makedirs(params.snapshotDir)

    # Load dataset
    train_dataset = ShapeNetDataset(
                    shapenet_dir="./data/shapenet/",
                    n_sample_points=10000,  # Match Michelangelo's training
                    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=params.batchSize, shuffle=True, num_workers=4
    )
    # test_dataset = R2N2ShapeNetDataset(
    #     partition="test",
    #     r2n2_shapenet_dir=params.modelsDataDir,
    #     synsets=params.synset,
    #     n_sample_points=params.nSamplePoints,
    # )
    # test_dataloader = DataLoader(
    #     test_dataset, batch_size=params.batchSize, shuffle=False, num_workers=4, collate_fn=train_dataset.collate_fn
    # )

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model

    netPred = PrimitiveTransformerQuaternion(
        n_primitives=params.nParts, # max seq len
        d_model=256,
        n_heads=4,
        n_layers=6,
        n_classes=len(params.primTypes)
    )
        
    if params.usePretrain:
        load_path = os.path.join(
            "./models/checkpoints",
            params.pretrainNet,
        )
        netPretrain = torch.load(load_path)
        netPred.load_state_dict(netPretrain)
        print("Loading pretrained model from {}".format(load_path))
        
    netPred.to(device)

    # Setup optimizer
    optimizer = torch.optim.Adam(netPred.parameters(), lr=params.learningRate)

    # Initialize training metrics

    # Train the model
    print("Iter\tErr\tTSDF\tChamf\tMeanRe")
    for iter in tqdm(range(params.numTrainIter), desc='Training progress'):
        loss, coverage, consistency, mean_reward = train(
            train_dataloader, netPred, optimizer, iter, params, device
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
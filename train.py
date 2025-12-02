"""
CUDA_VISIBLE_DEVICES=1 python cadAutoEncCuboids/primSelTsdfChamfer.py
"""

import os
import torch
from torch.utils.data import DataLoader
from models.prim_transformer import PrimitiveTransformerQuaternion
from dataloaders.shapenet import ShapeNetDataset
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

import modules.primitives as primitives
from losses import chamfer_distance_loss
from modules.config_utils import get_args
from utils.get_primitives import get_samples, get_primitives
from primitives.compose import generate_mesh_from_primitives
from visualization.render_mesh import render_mesh
from utils.get_optimizer import get_optimizer
from pytorch3d.structures import Meshes

torch.manual_seed(0)


def train(dataloader, netPred, optimizer, iter, params, device) -> float:
    # Get batch
    netPred.train()
    progress_bar = tqdm(dataloader, desc="Epoch progress", leave=False)
    for batch in progress_bar:
        sampledPoints, verts, faces = batch
        sampledPoints = sampledPoints.to(device)

        # scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
        # rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
        # translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
        # class_logits: (B, N_primitives, n_classes) - class logits
        # eos_logits: (B, N_primitives, 1) - end-of-sequence logits
        sequence = None
        log_probs = []
        point_feats = None
        for t in range(netPred.n_primitives):
            scale, rot, transl, cls, eos, point_feats = netPred(
                sequence=sequence, point_cloud=sampledPoints, point_features=point_feats
            )

            embedding = torch.cat([scale, rot, transl, eos, cls], dim=-1) # B, 1, 24

            sample, log_prob = get_samples(embedding) # B x 1 x 11
                
            sequence = sample if sequence is None else torch.concat([sequence, sample], dim=1) # (B, T + 1, 11)
            log_probs.append(log_prob)

        log_probs = torch.concat(log_probs, dim=1) # B x T x 1

        assert sequence is not None
        # Generate a mask that is only true if all previous sampled type where not the EOS token
        sampled_types = sequence[..., 10:] # B x T x 1
        mask = (sampled_types != 0).cumprod(dim=1) # B x T x 1
        # Mask out every primitive after the first EOS token
        log_probs = (log_probs * mask).sum(dim=[1, 2]) # B
        sequence = sequence * mask # B x T x 11

        primitives = get_primitives(sequence, netPred.n_primitives)
        vertices, faces = generate_mesh_from_primitives(primitives, device=device)

        # Start with max_loss for empty meshes
        B = sampledPoints.size(0)
        batch_loss = torch.full((B,), fill_value=1000, dtype=torch.float, device=device)

        # Mask out empty meshes
        empty_mask = (faces == -1).all(dim=[1, 2]) # B
        vertices = vertices[~empty_mask]
        faces = faces[~empty_mask]
        if len(faces) > 0:
            meshes = Meshes(
                verts=vertices, faces=faces
            )
            predPoints = sample_points_from_meshes(meshes, 10000)

            # cov_loss = coverage_loss(sampledPoints, predParts) # (B, N, 1)
            # cons_loss = consistency_loss(predParts, params.nSamplesChamfer, sampledPoints, inputVol) # (B, N, 1)
            # loss = cov_loss + params.chamferLossWt * cons_loss
            loss, _ = chamfer_distance_loss(
                predPoints, sampledPoints[:, :, :3], batch_reduction=None, point_reduction='mean'
            ) # B
            assert isinstance(loss, torch.Tensor)

            non_empty_indices = torch.arange(0, B, 1, device=device)[~empty_mask]
            batch_loss[non_empty_indices] = loss

        # Display metrics
        progress_bar.set_postfix_str(
            f"Total Loss: {batch_loss.mean().item():.4f}"
        )

        loss = (batch_loss * log_probs).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

@torch.inference_mode()
def evaluate(dataloader, netPred, device, epoch) -> float:
    # Get batch
    netPred.eval()
    progress_bar = tqdm(dataloader, desc="Validation progress", leave=False)
    visualization_count = 0
    validation_loss = 0
    n_batch = len(dataloader)
    for batch in progress_bar:
        sampledPoints, vertsGt, facesGt = batch
        sampledPoints = sampledPoints.to(device)
        vertsGt = vertsGt.to(device)
        facesGt = facesGt.to(device)

        # scale_params: (B, N_primitives, 6) - μ and σ for 3D scale
        # rotation_params: (B, N_primitives, 8) - μ and σ for quaternion
        # translation_params: (B, N_primitives, 6) - μ and σ for 3D translation
        # class_logits: (B, N_primitives, n_classes) - class logits
        # eos_logits: (B, N_primitives, 1) - end-of-sequence logits
        sequence = None
        point_feats = None
        for t in range(netPred.n_primitives):
            scale, rot, transl, cls, eos, point_feats = netPred(
                sequence=sequence, point_cloud=sampledPoints, point_features=point_feats
            )

            embedding = torch.cat([scale, rot, transl, eos, cls], dim=-1)

            if sequence is None or sequence[-1][:, t-1, 0] != 0:
                sample, _ = get_samples(embedding) # B x 1 x 11
            else:
                sample, _ = torch.zeros_like(sequence[:, :1, :]), torch.zeros_like(log_probs[-1])
                
            sequence = sample if sequence is None else torch.concat([sequence, sample], dim=1) # (B, T + 1, 11)

        primitives = get_primitives(sequence, netPred.n_primitives)
        vertices, faces = generate_mesh_from_primitives(primitives)
        meshes = Meshes(
            verts=vertices, faces=faces
        )
        predPoints = sample_points_from_meshes(meshes, 10000)
        loss, _ = chamfer_distance_loss(predPoints, sampledPoints, point_reduction='mean')
        assert isinstance(loss, torch.Tensor)

        # Visualize predicted mesh
        for index in range(len(vertices)):
            output_dir = f'visualizations/epoch_{epoch}/'
            output_filename_format = '{:d}.gif'.format
            output_filename_format_gt = '{:d}_gt.gif'.format
            render_mesh(vertices[index], faces[index], output_dir + output_filename_format(visualization_count), device=device)
            render_mesh(vertsGt[index], facesGt[index], output_dir + output_filename_format_gt(visualization_count), device=device)
            visualization_count +=1

        validation_loss += loss.item() / n_batch

    return validation_loss

def main():

    params = get_args()
    params.visDir = os.path.join("output/visualization/", params.name)
    params.visMeshesDir = os.path.join("output/visualization/meshes/", params.name)
    params.snapshotDir = os.path.join("output/snapshots/", params.name)
    params.primTypes = 3 # TODO Change to CLI

    if not os.path.exists(params.visDir):
        os.makedirs(params.visDir)

    if not os.path.exists(params.visMeshesDir):
        os.makedirs(params.visMeshesDir)

    if not os.path.exists(params.snapshotDir):
        os.makedirs(params.snapshotDir)

    # Load dataset
    train_dataset = ShapeNetDataset(
        shapenet_dir="./data/shapenet_train/",
        n_sample_points=10000,  # Match Michelangelo's training
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=params.batchSize, shuffle=True, num_workers=4
    )
    test_dataset = ShapeNetDataset(
        shapenet_dir="./data/shapenet_test/",
        n_sample_points=10000,  # Match Michelangelo's training
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=params.batchSize, shuffle=False, num_workers=4
    )

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize model

    netPred = PrimitiveTransformerQuaternion(
        n_primitives=params.nParts, # max seq len
        d_model=256,
        n_heads=4,
        n_layers=6,
        n_classes=params.primTypes
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
    optimizer = get_optimizer(netPred)

    # Initialize training metrics

    # Train the model
    for iter in tqdm(range(params.numTrainIter), desc='Training progress'):
        loss = train(
            train_dataloader, netPred, optimizer, iter, params, device
        )

        # Visualize results
        if iter % params.visIter == 0:
            evaluate(test_dataloader, netPred, device, epoch=iter)

        if ((iter + 1) % 10) == 0:
            torch.save(
                netPred.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter)
            )

if __name__ == '__main__':
    main()
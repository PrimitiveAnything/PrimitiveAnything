import torch
import torch.nn.functional as F

from modules.transformer import rigidTsdf


def coverage_loss(sampledPoints, predParts):  ## coverage loss
    """
    To what degree is the ground truth model inside the predicted composition

    Returns the truncated (always positive) signed distance between the
    sampled points and the surface and the composition.

    :param sampledPoints: Points in the surface of the ground truth model
    :param predParts: Predicted parts of the composition
    """
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
    # Get the min coverage loss across parts
    tsdf_final = -1 * F.max_pool1d(-1 * tsdf_all, kernel_size=nParts)  # B x nP
    tsdf_final = tsdf_final.mean()
    return tsdf_final


def tsdf_transform(sample_points, part):
    ## sample_points Batch_size x nP x 2, # parts Batch_size x 1 x 10
    shape = part[:, :, 0:3]  # B x 1 x 3
    trans = part[:, :, 3:6]  # B  x 1 x 3
    quat = part[:, :, 6:10]  # B x 1 x 4

    p1 = rigidTsdf(sample_points, trans, quat)  # B x nP x 3
    tsdf = cuboid_tsdf(p1, shape)  # B x nP x 1
    return tsdf


def cuboid_tsdf(sample_points, shape):
    ## sample_points Batch_size x nP x 3 , shape Batch_size x 1 x 3,
    ## output Batch_size x nP x 3
    nP = sample_points.size(1)
    shape_rep = shape.repeat(1, nP, 1)
    tsdf = torch.abs(sample_points) - shape_rep
    tsdfSq = F.relu(tsdf).pow(2).sum(dim=2, keepdim=True)
    return tsdfSq  ## Batch_size x nP x 1


def get_existence_weights(tsdf, part):
    e = part[:, :, 11:12]
    e = e.expand(tsdf.size())
    e = (1 - e) * 10
    return e

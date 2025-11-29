import torch
import torch.nn.functional as F
from primitives import CuboidSurface, EllipsoidSurface, EllipticalCylinderSurface
from .sample import get_sample, get_sample_and_probs

def get_samples(embedding):
    """
    Takes in the transformer embedding of dim B x 1 x 23 and outputs
    a B x 1 x 11 tensor (scale, rotation, translation, class)
    """
    B, nPart, _ = embedding.shape
    scale_mean, scale_std = embedding[:, :, 0:3], embedding[:, :, 3:6]
    rot_mean, rot_std = embedding[:, :, 6:10], embedding[:, :, 10:14]
    trans_mean, trans_std = embedding[:, :, 14:17], embedding[:, :, 17:20]
    probs = F.softmax(embedding[:, :, 20:], dim=-1)

    next_state = torch.empty((B, nPart, 11), dtype=torch.float64, device=embedding.device)

    scale = get_sample(scale_mean, scale_std)
    rotation = get_sample(rot_mean, rot_std)
    translation = get_sample(trans_mean, trans_std)

    next_state[:, :, 0:3] = scale 
    next_state[:, :, 3:7] = rotation 
    next_state[:, :, 7:10] = translation 
    next_state[:, :, 10] = torch.argmax(probs, dim=-1)

    return next_state

def get_primitives(samples):
    """
    Input: samples, B x T x 11
    Output: List[List[Primitives]]
    """
    B, T, _ = samples.shape

    scale = samples[:, :, :3]
    rotation = samples[:, :, 3:7]
    translation = samples[:, :, 7:10]
    prim_type = samples[:, :, 10].long()

    all_batches = []

    for b in range(B):
        batch_list = []
        for t in range(T):
            prim = prim_type[b, t].item()

            s = scale[b, t]
            r = rotation[b, t]
            tr = translation[b, t]

            if prim == 0:
                batch_list.append(CuboidSurface(s, r, tr))
            elif prim == 1:
                batch_list.append(EllipsoidSurface(s, r, tr))
            else:
                batch_list.append(EllipticalCylinderSurface(s, r, tr))

        all_batches.append(batch_list)
    
    return all_batches
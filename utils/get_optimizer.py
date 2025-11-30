import torch
from models.primitive_anything.michelangelo import ShapeConditioner as ShapeConditioner_miche

def get_optimizer(model, lr=1e-4, weight_decay=0.01):
    params = []

    for name, module in model.named_modules():
        if isinstance(module, ShapeConditioner_miche):
            continue
            
        for name, par in module.named_parameters(recurse=False):
            params.append(par)
        
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    return optimizer
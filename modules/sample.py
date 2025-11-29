import torch

def get_probs(mean, std_dev, sample):
    var = std_dev ** 2
    coeff = 1.0 /(torch.sqrt(2 * torch.pi * var))

    exponent = -0.5 * ((sample - mean) ** 2) / var
    return coeff * torch.exp(exponent)

def get_log_probs(mean, std_dev, sample):
    var = std_dev ** 2
    return -0.5 * (torch.log(2 * torch.pi * var) +
                   ((sample - mean) ** 2) / var)

def get_sample(mean, std_dev):
    eps = torch.randn_like(mean)
    sample = mean + eps * std_dev
    return sample

def get_sample_and_probs(mean, std_dev):
    sample = get_sample
    probs = get_probs(mean, std_dev, sample)
    log_probs = get_log_probs(mean, std_dev, sample)
    return sample, probs, log_probs
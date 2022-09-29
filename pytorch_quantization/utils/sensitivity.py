import torch
import numpy as np

def torch_snr_error(pred: torch.Tensor, real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    if type(pred) is np.ndarray:
        pred = torch.from_numpy(pred)
    if type(real) is np.ndarray:
        real = torch.from_numpy(real)

    if pred.shape != real.shape:
        raise ValueError(f'Can not compute snr loss for tensors with different shape. '
            f'({pred.shape} and {real.shape})')
        
    reduction = str(reduction).lower()

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        real = real.unsqueeze(0)

    pred = pred.flatten(start_dim = 1)
    real = real.flatten(start_dim = 1)

    noise_power = torch.pow(pred - real, 2).sum(dim=-1)
    signal_power = torch.pow(real, 2).sum(dim=-1)
    snr = (noise_power) / (signal_power + 1e-7)

    if reduction == 'mean':
        return torch.mean(snr)
    elif reduction == 'sum':
        return torch.sum(snr)
    elif reduction == 'none':
        return snr
    else:
        raise ValueError(f'Unsupported reduction method.')

def torch_mse_error(pred: torch.Tensor, real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    if type(pred) is np.ndarray:
        pred = torch.from_numpy(pred)
    if type(real) is np.ndarray:
        real = torch.from_numpy(real)
        
    if pred.shape != real.shape:
        raise ValueError(f'Can not compute mse loss for tensors with different shape. '
            f'({pred.shape} and {real.shape})')
    
    reduction = str(reduction).lower() 

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        real = real.unsqueeze(0)

    pred = pred.flatten(start_dim = 1)
    real = real.flatten(start_dim = 1)

    mse = torch.pow(pred - real, 2)
    
    if reduction == 'mean':
        return torch.mean(mse)
    elif reduction == 'sum':
        return torch.sum(mse)
    elif reduction == 'none':
        return mse
    else:
        raise ValueError(f'Unsupported reduction method.')

def torch_cosine_error(pred: torch.Tensor, real: torch.Tensor, reduction: str='mean') -> torch.Tensor:
    if type(pred) is np.ndarray:
        pred = torch.from_numpy(pred)
    if type(real) is np.ndarray:
        real = torch.from_numpy(real)

    if pred.shape != real.shape:
        raise ValueError(f'Can not compute cosine loss for tensors with different shape. '
            f'({pred.shape} and {real.shape})')
    
    reduction = str(reduction).lower() 

    if pred.ndim == 1:
        pred = pred.unsqueeze(0)
        real = real.unsqueeze(0)

    pred = pred.flatten(start_dim = 1)
    real = real.flatten(start_dim = 1)

    cosine = torch.cosine_similarity(pred, real, dim=1)

    if reduction == 'mean':
        return 1 - torch.mean(cosine)
    elif reduction == 'sum':
        return torch.sum(1 - cosine)
    elif reduction == 'none':
        return 1 - cosine
    else:
        raise ValueError(f'Unsupported reduction method.')






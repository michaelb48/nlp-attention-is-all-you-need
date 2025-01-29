import numpy as np
import re
import os
import copy
import random
from itertools import islice
import gc
import torch
import torch.nn.functional as F

# copied from lecture notebook
def set_seed(seed: int):
    # Set seed for Python's random module
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch
    torch.manual_seed(seed)
    
    # Set seed for CUDA (if using)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        
    # Make PyTorch deterministic (this can slow down the computation)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_directory_exists(path:str):
    directory = os.path.dirname(os.path.abspath(path))  # Convert to absolute path
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

def average_model_weights(state_dicts):
    """
    Averages the weights from multiple state_dicts.

    Args:
        state_dicts: List of state_dict dictionaries

    Returns:
        Averaged state_dict.
    """
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        for state_dict in state_dicts[1:]:
            avg_state_dict[key] += state_dict[key]
        avg_state_dict[key] /= len(state_dicts)
    return avg_state_dict

def save_checkpoint(model, optimizer, epoch, save_dir,step=None, config_file=None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if step is not None:
        checkpoint_path = os.path.join(save_dir, f'{config_file}_model_epoch_{epoch}_step_{step}.pth')
    elif step is None and config_file is not None:
        checkpoint_path = os.path.join(save_dir, f'{config_file}_model_epoch_{epoch}.pth')
    else:
        checkpoint_path = os.path.join(save_dir, f'model_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} to {checkpoint_path}")
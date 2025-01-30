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

def get_model_paths(path_to_results, model_name_starts_with):
    """
    Gets the paths of the models that start with "model_name_starts_with"

    Args:
        path_to_results (str): The path to the folder containing the results.
        model_name_starts_with (str): The name of the model that starts with.

    Returns:
        A list of paths to the model files.
    """
    all_files = os.listdir(path_to_results)
    model_files = [file for file in all_files if file.startswith(model_name_starts_with)]
    model_files = [os.path.join(path_to_results, filename) for filename in model_files]
    print(f"Loading {len(model_files)} files from model:")
    for file in model_files:
        print(file)
    print("")
    return model_files

def load_state_dicts(file_paths):
    """
    Loads the state dictionaries of the given file paths

    Args:
        file_paths (str): The paths to the models.

    Returns:
        A list of state dictionaries.
    """
    state_dicts = []
    print("Loading model state dicts:")
    for file in file_paths:
        loaded_dict = torch.load(file, map_location="cpu")
        if "model_state_dict" in loaded_dict:
            print(f"Extracting 'model_state_dict' from checkpoint in {file}")
            state_dicts.append(loaded_dict["model_state_dict"])
    print("")
    return state_dicts

def average_model_weights(state_dicts):
    """
    Averages the weights from multiple state_dicts.

    Args:
        state_dicts: List of state_dict dictionaries

    Returns:
        Averaged state_dict.
    """
    avg_state_dict = copy.deepcopy(state_dicts[0])
    print("Averaging model...")
    for key in avg_state_dict.keys():
        for state_dict in state_dicts[1:]:
            avg_state_dict[key] += state_dict[key]
        avg_state_dict[key] /= len(state_dicts)
    return avg_state_dict

def save_avg_model_weights(save_path, save_name, avg_state_dict):
    """
    Saves the averaged state_dicts of all model checkpoints in the given path.

    Args:
        save_path (str): The path to save the averaged state_dicts.
        save_name (str): The name of the averaged state_dicts.
        avg_state_dict (dict): Averaged state_dicts.
    """
    final_save_path = os.path.join(save_path, save_name)
    torch.save(avg_state_dict, final_save_path)
    print(f"Averaged model saved to {final_save_path}\n")

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

def load_checkpoint(file_path, model, device='cpu'):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"No checkpoint found at '{file_path}'")

    checkpoint = torch.load(file_path, map_location=device)
    
    print(f"Loaded checkpoint from '{file_path}'")
    # Use the state_dict of the loaded model
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print("Model correctly loaded.")

    return model

import os
import yaml
import torch


# This utility function helps to load the configuration file in YAML format
def load_config(config_path="config/config.yaml"):

    # Check if the specified configuration file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Open the configuration file and parse its contents as a Python dictionary
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


#  This function is a utility to check if a given directory exists
def check_if_directory_exists(dir_path):

    # If the directory exists, return True; otherwise, return False
    if os.path.exists(dir_path):
        return True

    return False


# In this utility function we check if a given directory is empty
def is_directory_empty(dir_path):

    # If the directory contains any items (files or subdirectories), return False (not empty)
    if any(os.scandir(dir_path)):
        return False

    return True


# This utility function helps to count the number of files in a
# given directory (excluding subdirectories)
def count_files(directory):
    return len(
        [
            file
            for file in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, file))
        ]
    )


# It helps to access a specific key from a nested configuration dictionary
def get_config_key(config, key):

    # Split the key into individual components (by ".")
    keys = key.split(".")
    current_dict = config

    try:
        # Traverse the nested dictionary using the split keys
        for k in keys:
            if k in current_dict:

                # Move to the next level in the nested dictionary
                current_dict = current_dict[k]
            else:
                raise KeyError(f"Key '{key}' not found in the configuration.")
        return current_dict
    except Exception as e:
        raise KeyError(f"Error accessing '{key}': {str(e)}")


# This utility function helps in selecting the appropriate
# device (GPU, MPS, or CPU) for model training/inference
def select_device():
    # Check if a CUDA-compatible GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")

    # Check if Apple Silicon MPS (Metal Performance Shaders) is available (for Mac M1/M2 chips)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # If no GPU or MPS device is available, default to CPU
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    return device

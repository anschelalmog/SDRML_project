import numpy as np
import random
import torch
import matplotlib.pyplot as plt


def set_global_seed(seed):
    """
    Ensure reproducibility across runs by setting random seeds.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

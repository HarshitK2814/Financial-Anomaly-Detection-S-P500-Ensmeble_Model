import random
import numpy as np

def set_seed(seed: int):
    """
    Set the random seed for reproducibility.

    Parameters:
    seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
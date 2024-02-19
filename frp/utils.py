from typing import Union
import numpy as np
import torch
import igraph as ig


def check_numpy(vars):
    for cnt, var in enumerate(vars):
        if not isinstance(var, np.ndarray):
            raise TypeError(f"{cnt} th variable should be a numpy array.")


def require_dict(d, keys):
    missing_keys = [key for key in keys if key not in d]
    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys: {missing_keys}")

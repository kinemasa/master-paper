# dataloaders.py
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split

def make_loaders(dataset, batch_size:int, num_workers:int,
                 train_ratio:float, val_ratio:float) -> Tuple[DataLoader, DataLoader, DataLoader]:
    N = len(dataset)
    n_train = int(N * train_ratio)
    n_val   = int(N * val_ratio)
    n_test  = N - n_train - n_val
    g = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test], generator=g)
    common_kwargs = dict(num_workers=num_workers, pin_memory=True,
                         drop_last=False, persistent_workers=(num_workers > 0), prefetch_factor=4)
    dl_train = DataLoader(train_set, batch_size=batch_size, shuffle=True,  **common_kwargs)
    dl_val   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, **common_kwargs)
    dl_test  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, **common_kwargs)
    return dl_train, dl_val, dl_test
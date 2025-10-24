# dataloaders.py
from typing import Tuple
import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.data import DataLoader, Subset
import numpy as np
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


def make_fold_indices(n_samples: int, n_splits: int = 5, seed: int = 42):
    """
    n_samplesをn_splitsに分割し、(train_idx, test_idx) のリストを返す。
    各foldのtestは約20%。残り80%をtrainとして返す（valは別途train内で切り出す）。
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_samples)
    folds = np.array_split(order, n_splits)
    pairs = []
    all_idx = np.arange(n_samples)
    for k in range(n_splits):
        test_idx = np.sort(folds[k])
        mask = np.ones(n_samples, dtype=bool)
        mask[test_idx] = False
        train_idx = np.sort(all_idx[mask])
        pairs.append((train_idx, test_idx))
    return pairs


def split_train_val(train_idx: np.ndarray, val_ratio: float = 0.20, seed: int = 42):
    """
    train部分(=全体の80%)の中から、さらにvalを取り出す（ReduceLROnPlateau用）。
    例: val_ratio=0.2 → 全体に対して train64% / val16% / test20% 程度。
    """
    if val_ratio <= 0:
        return train_idx, np.array([], dtype=int)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(train_idx)
    n_val = int(len(train_idx) * val_ratio)
    val_idx = np.sort(perm[:n_val])
    tr_idx = np.sort(perm[n_val:])
    return tr_idx, val_idx


def make_loaders_from_indices(dataset,
                              train_idx, val_idx, test_idx,
                              batch_size=32, num_workers=2, shuffle_train=True):
    ds_train = Subset(dataset, train_idx.tolist())
    ds_val   = Subset(dataset, val_idx.tolist()) if len(val_idx) > 0 else Subset(dataset, [])
    ds_test  = Subset(dataset, test_idx.tolist())

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=shuffle_train,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_val   = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    dl_test  = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, pin_memory=True, drop_last=False)
    return dl_train, dl_val, dl_test
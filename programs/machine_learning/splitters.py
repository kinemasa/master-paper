
import os, glob
import numpy as np
import pandas as pd
import joblib, optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, KFold
import lightgbm as lgb

def get_loso_splits(df, groups):
    """被験者LOSO: sklearnのsplitジェネレータを返す"""
    n = groups.nunique()
    return GroupKFold(n_splits=n).split(df, groups=groups)

def get_kfold_splits(df, n_splits, seed=42):
    """ランダムk-fold: sklearnのsplitジェネレータを返す"""
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(df)


def get_subject_kfold_splits(df, n_splits, seed=42):
    """被験者単位k-fold: リスト[(train_idx, test_idx), ...]を返す"""
    subjects = df["subject_id"].values
    uniq = np.array(sorted(df["subject_id"].unique()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    for tr_sub_i, te_sub_i in kf.split(uniq):
        te_sub = set(uniq[te_sub_i])
        mask = np.isin(subjects, list(te_sub))
        test_idx = np.where(mask)[0]
        train_idx = np.where(~mask)[0]
        splits.append((train_idx, test_idx))
    return splits

def get_within_loto_splits(df):
    """被験者内LOTO: リスト[(train_idx, test_idx), ...]を返す"""
    splits = []
    for _, idx in df.groupby("subject_id").groups.items():
        idx = np.array(list(idx))
        trials = df.loc[idx, "trial_id"].values
        for t in np.unique(trials):
            test_mask = (trials == t)
            test_idx = idx[test_mask]
            train_idx = idx[~test_mask]
            if len(test_idx) and len(train_idx):
                splits.append((train_idx, test_idx))
    return splits

def iter_spliter(df, groups, cv_method, n_splits=5, seed=42):
    """
    どの方式でも“反復可能”に正規化して返すラッパ。
    呼び出し側は for tr, te in iter_splits(...): で統一使用。
    """
    if cv_method == "loso":
        yield from get_loso_splits(df, groups)                      # ジェネレータをそのまま委譲
    elif cv_method == "kfold":
        yield from get_kfold_splits(df, n_splits, seed)
    elif cv_method == "subject_kfold":
        for tr, te in get_subject_kfold_splits(df, n_splits, seed): # リスト→反復
            yield tr, te
    elif cv_method == "within_loto":
        for tr, te in get_within_loto_splits(df):
            yield tr, te
    else:
        raise ValueError("cv_method は 'loso' | 'kfold' | 'subject_kfold' | 'within_loto'")

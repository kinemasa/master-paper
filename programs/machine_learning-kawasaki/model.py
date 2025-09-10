import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from lightgbm import LGBMRegressor
import optuna

def objective_rf(trial, X, y,SEED):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 200),
        max_depth=trial.suggest_int("max_depth", 2, 32),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        random_state=SEED,
        n_jobs=-1,
    )
    model = RandomForestRegressor(**params)
    score = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=SEED),
                            scoring="neg_mean_absolute_error", n_jobs=-1)
    return -float(score.mean())

def objective_svr(trial, X, y) :
    C = trial.suggest_float("C", 1e-3, 1e3, log=True)
    epsilon = trial.suggest_float("epsilon", 1e-3, 1e1, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
    model = SVR(C=C, epsilon=epsilon, kernel=kernel, degree=degree)
    score = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=SEED),
                            scoring="neg_mean_absolute_error", n_jobs=-1)
    return -float(score.mean())

def objective_lgbm(trial, X, y,SEED):
    params = dict(
        num_leaves=trial.suggest_int("num_leaves", 2, 256),
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        learning_rate=trial.suggest_float("learning_rate", 1e-3, 1.0, log=True),
        feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.4, 1.0),
        bagging_freq=trial.suggest_int("bagging_freq", 1, 7),
        random_state=SEED,
        n_jobs=-1,
    )
    model = LGBMRegressor(**params)
    score = cross_val_score(model, X, y, cv=KFold(5, shuffle=True, random_state=SEED),
                            scoring="neg_mean_absolute_error", n_jobs=-1)
    return -float(score.mean())

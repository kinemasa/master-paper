from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import lightgbm as lgb


def make_model(name, RANDOM_SEED, params=None):
    params = params or {}
    if name == "rf":
        model = RandomForestRegressor(
        n_estimators=params.get("n_estimators", 300),
        max_depth=params.get("max_depth", None),
        random_state=RANDOM_SEED,
        n_jobs=-1)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])
    if name == "lgbm":
        model = lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 300),
        learning_rate=params.get("learning_rate", 0.05),
        max_depth=params.get("max_depth", -1),
        random_state=RANDOM_SEED,
        n_jobs=-1)
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])
    if name == "svr":
        model = SVR(
        C=params.get("C", 1.0),
        epsilon=params.get("epsilon", 0.1),
        kernel=params.get("kernel", "rbf"))
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", model)])
    raise ValueError("MODEL_NAME は 'rf','lgbm','svr' を指定してください")
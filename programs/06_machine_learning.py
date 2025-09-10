import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys
import optuna
import dtale
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from machine_learning.load_and_save import subject_to_id,subject_to_trial,read_features,read_labels,merge_data,pick_feature_cols
from machine_learning.evaluation import aami_metrics,bhs_grade
from machine_learning.make_model import make_model
from machine_learning.splitters import iter_spliter,get_kfold_splits

from myutils.select_folder import select_folder,select_file
# =============== 基本設定 ===============
# ======================== 設定 ========================
FEATURES_DIR = select_folder(message="特徴量の入ったフォルダを選択")
BP_MEASURED_CSV = select_file(message="血圧値の正解値ファイル")
MODEL_NAME = "rf" # "rf", "lgbm", "svr"
CV_METHOD = "loso" # "loso" | "kfold" | "subject_kfold"（被験者をk分割） | "within_loto"（被験者内で試行LOTO）| "kfold" | "within_loto"（被験者内で試行LOTO）
N_SPLITS = 5 # k-foldの分割数（CV_METHODがkfoldのとき有効）
SAVE_DIR = "./models"


TRAIN_NEW = True # True: 新規学習 / False: 保存済みモデルで評価のみ
USE_OPTUNA = True
DEBUG_CV = True # デバッグ出力のON/OFF（不要なら False）
OPTUNA_TRIALS = 5
RANDOM_SEED = 42
# ======================================================

# --- RFEで選択する特徴量数 ---
N_FEATURES_TO_SELECT = 25

warnings.filterwarnings("ignore", category=UserWarning)

def tune_with_optuna(df, X, y, groups, model_name, n_trials=40):
    def objective(trial):
        if model_name == "rf":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 3, 30),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            }
        elif model_name == "lgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            }
        elif model_name == "svr":
            params = {
                "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
                "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
            }
        else:
            raise ValueError("MODEL_NAME must be rf, lgbm, or svr")

        # OOF予測してMAEを評価
        oof = np.zeros(len(y))
        for tr, te in iter_spliter(df, groups,CV_METHOD):
            model = make_model(model_name,RANDOM_SEED, params)
            model.fit(X.iloc[tr], y.iloc[tr])
            oof[te] = model.predict(X.iloc[te])
        return np.mean(np.abs(oof - y.values))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def main():
    # === データ読み込み ===
    feat = read_features(FEATURES_DIR)
    lab  = read_labels(BP_MEASURED_CSV)
    df = feat.merge(lab, on=["subject_id"], how="inner")
    df = df.drop_duplicates(subset=["subject_id", "trial"])
    df.to_csv("merged.csv", index=False, encoding="utf-8-sig")
    X = df[pick_feature_cols(df)]
    y_sbp = df["SBP"].astype(float)
    y_dbp = df["DBP"].astype(float)
    groups = df["subject_id"]

    # === 既存モデルで評価のみ ===
    if not TRAIN_NEW:
        sbp_pack = joblib.load(os.path.join(SAVE_DIR, "sbp_model.joblib"))
        dbp_pack = joblib.load(os.path.join(SAVE_DIR, "dbp_model.joblib"))
        sbp_model = sbp_pack["model"]
        dbp_model = dbp_pack["model"]
        sbp_best_params = sbp_pack.get("best_params", {})
        dbp_best_params = dbp_pack.get("best_params", {})

        print("保存済みモデルをロードして評価します…")
        sbp_pred = sbp_model.predict(X)
        dbp_pred = dbp_model.predict(X)
        print("SBP AAMI:", aami_metrics(y_sbp, sbp_pred), "BHS:", bhs_grade(y_sbp, sbp_pred))
        print("DBP AAMI:", aami_metrics(y_dbp, dbp_pred), "BHS:", bhs_grade(y_dbp, dbp_pred))
        return

    # === Optunaによるチューニング ===
    if USE_OPTUNA:
        print("OptunaでSBPのパラメータ探索中...")
        sbp_best_params = tune_with_optuna(df, X, y_sbp, groups, MODEL_NAME, OPTUNA_TRIALS)
        print("OptunaでDBPのパラメータ探索中...")
        dbp_best_params = tune_with_optuna(df, X, y_dbp, groups, MODEL_NAME, OPTUNA_TRIALS)
    else:
        sbp_best_params, dbp_best_params = {}, {}

    print(f"モデル={MODEL_NAME}, 検証={CV_METHOD}")

    # === OOF入れ物 ===
    sbp_oof = np.zeros(len(y_sbp))
    dbp_oof = np.zeros(len(y_dbp))
    
    # 各行がどのfoldでテストになったかを記録（-1=未割当）
    df["cv_fold"] = -1
    
    fold = 0


    # === 各分割で学習・予測 ===
    for tr, te in iter_spliter(df, groups,CV_METHOD):
        fold +=1
        if DEBUG_CV:
            # 被験者IDの一覧を表示（漏洩チェック用に train/test の被験者集合も見る）
            tr_subjects = df.iloc[tr]["subject_id"].unique()
            te_subjects = df.iloc[te]["subject_id"].unique()

            print(f"\n=== Fold {fold} ===")
            print(f"train 行数: {len(tr)} / test 行数: {len(te)}")
            print("train 被験者:", list(tr_subjects))
            print("test  被験者:", list(te_subjects))

            # 漏洩チェック（同一被験者が同一foldで train と test の両方に入っていないか）
            leak = set(tr_subjects).intersection(set(te_subjects))
            if len(leak) > 0:
                print("⚠️  同一fold内で被験者が重複しています（データリーク）:", leak)

            # 必要なら試行の内訳も表示（trial列がある場合）
            if "trial" in df.columns:
                te_detail = (df.iloc[te][["subject_id", "trial"]]
                            .value_counts()
                            .reset_index(name="n"))
                print("test 内訳（subject_id, trial, 件数）:\n", te_detail.head(10))

            # テスト行にfold番号をマーキング（あとで集計に便利）
            df.loc[df.index[te], "cv_fold"] = fold
        
        ms = make_model(MODEL_NAME,RANDOM_SEED, sbp_best_params)
        md = make_model(MODEL_NAME,RANDOM_SEED, dbp_best_params)
        # SBP
        ms.fit(X.iloc[tr], y_sbp.iloc[tr])
        sbp_oof[te] = ms.predict(X.iloc[te])
        # DBP
        md.fit(X.iloc[tr], y_dbp.iloc[tr])
        dbp_oof[te] = md.predict(X.iloc[te])

    # === 評価（AAMI/BHS） ===
    print("SBP AAMI:", aami_metrics(y_sbp, sbp_oof), "BHS:", bhs_grade(y_sbp, sbp_oof))
    print("DBP AAMI:", aami_metrics(y_dbp, dbp_oof), "BHS:", bhs_grade(y_dbp, dbp_oof))

    # === 全データで最終学習 & 保存 ===
    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        final_sbp = make_model(MODEL_NAME,RANDOM_SEED, sbp_best_params)
        final_dbp = make_model(MODEL_NAME,RANDOM_SEED, dbp_best_params)
        final_sbp.fit(X, y_sbp)
        final_dbp.fit(X, y_dbp)
        joblib.dump({"model": final_sbp, "best_params": sbp_best_params}, os.path.join(SAVE_DIR, "sbp_model.joblib"))
        joblib.dump({"model": final_dbp, "best_params": dbp_best_params}, os.path.join(SAVE_DIR, "dbp_model.joblib"))
        print("モデルとベストパラメータを保存しました")

if __name__ == "__main__":
    
    main()

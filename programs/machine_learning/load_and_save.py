import os, glob
import pandas as pd



# ----------------- データ読み込み -----------------
def subject_to_id(s):
    # 先頭の 被験者の名前を取り出して返す関数 
    return str(s).split("-")[0]

def subject_to_trial(s):
    # 先頭の 被験者の名前を取り出して返す関数 
    return str(s).split("-")[1]


def read_features(dir_path):
    # 指定ディレクトリ内のCSVファイルをすべて読み込み
    paths = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["subject_id"] = df["subject"].map(subject_to_id) ## 被験者のidを登録する。
        # df["trial"]      = df["subject"].map(subject_to_trial) ## 被験者のtrialを登録する。
        
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def read_labels(csv_path):
    # 血圧ラベルのCSVを読み込み
    # 想定フォーマット: subject, trial, SBP, DBP
    lab = pd.read_csv(csv_path)
    # 列名を統一: subject → subject_id
    if "subject_id" not in lab.columns:
        lab = lab.rename(columns={"subject": "subject_id"})

    # # trial列がある場合も保持しておく（解析時に使えるように）
    # if "trial" in lab.columns:
    #     return lab[["subject_id", "trial", "SBP", "DBP","age"]]

    # trial列がない場合は従来通り
    return lab[["subject_id", "trial", "SBP", "DBP","age"]]


def merge_data(feat, lab):
    # 特徴量DataFrameとラベルDataFrameを subject_id で内部結合する
    # → 両方に存在する subject_id のデータだけが残る
    return feat.merge(lab, on="subject_id", how="inner")


def pick_feature_cols(df):
    drop = {"SBP","DBP","subject_id","subject","method","roi"}
    return [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
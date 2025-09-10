import os
import joblib
import pandas as pd


def save_text(path, text) :
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(text)

def save_model(model, model_name, target, select,save_folder) -> None:
    fname = f"{model_name}_{target}_{select}.pkl".replace(" ", "_")
    fpath = os.path.join(save_folder, fname)
    joblib.dump(model, fpath)

def save_selected_features(model_name, target, selected_features,save_folder) -> None:
    path = os.path.join(save_folder, "selected_features.txt")
    save_text(path, f"Model: {model_name} | Target: {target}\n"
                    f"Selected indices: {selected_features}\n\n")

def save_best_params(best_params,save_folder):
    path = os.path.join(save_folder, "best_params.txt")
    lines = ["Camera Type:1\n"]
    for k, v in best_params.items():
        lines.append(f"{k}:")
        for pk, pv in v.items():
            lines.append(f"  {pk}: {pv}")
        lines.append("")
    save_text(path, "\n".join(lines) + "\n")
    
def save_subject_results(subject_results,save_folder) -> None:
    df = pd.DataFrame(subject_results)
    fpath = os.path.join(save_folder, f"subject_results.csv".replace(" ", "_"))
    df.to_csv(fpath, index=False, encoding="utf-8-sig")
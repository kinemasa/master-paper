import numpy as np

def aami_metrics(y_true, y_pred):
    err = y_pred - y_true
    return {"mean_error": float(np.mean(err)), "sd": float(np.std(err, ddof=1)), "mae": float(np.mean(np.abs(err)))}


def bhs_grade(y_true, y_pred):
    ae = np.abs(y_pred - y_true)
    p5, p10, p15 = 100*np.mean(ae<=5), 100*np.mean(ae<=10), 100*np.mean(ae<=15)
    if (p5>=60) and (p10>=85) and (p15>=95): grade="A"
    elif (p5>=50) and (p10>=75) and (p15>=90): grade="B"
    elif (p5>=40) and (p10>=65) and (p15>=85): grade="C"
    else: grade="D"
    return {"p5":p5,"p10":p10,"p15":p15,"grade":grade}
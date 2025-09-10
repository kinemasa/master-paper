import numpy as np



# =============== AAMI/BHS 判定 ===============
def determine_aami_standard(diff):
    """
    AAMI: |ME| <= 5 mmHg かつ SD <= 8 mmHg
    diffは「参照 − 推定」のベクトル
    """
    me = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    ok = (abs(me) <= 5.0) and (sd <= 8.0)
    return ok

def determine_bhs_grade(diff: np.ndarray):
    """
    BHS成績：<5, <10, <15 mmHg の割合とグレード
    閾値：A(0.60/0.85/0.95), B(0.50/0.75/0.90), C(0.40/0.65/0.85)
    """
    adiff = np.abs(diff)
    r5 = float(np.mean(adiff < 5))
    r10 = float(np.mean(adiff < 10))
    r15 = float(np.mean(adiff < 15))
    grade = "D"
    if (r5 >= 0.60) and (r10 >= 0.85) and (r15 >= 0.95):
        grade = "A"
    elif (r5 >= 0.50) and (r10 >= 0.75) and (r15 >= 0.90):
        grade = "B"
    elif (r5 >= 0.40) and (r10 >= 0.65) and (r15 >= 0.85):
        grade = "C"
    return r5, r10, r15, grade
import matplotlib.pyplot as plt

# データ
labels = ["5", "10", "60"]  # 横軸ラベルをカテゴリとして扱う
x = range(len(labels))      # 等間隔 [0,1,2]
sbp_error = [12.84, 11.98, 11.85]
dbp_error = [8.10, 6.69, 6.67]

# プロット
plt.figure(figsize=(6, 4))
plt.plot(x, sbp_error, marker="o", markersize=8, linewidth=2, label="SBP")
plt.plot(x, dbp_error, marker="s", markersize=8, linewidth=2, label="DBP")


# 軸ラベルとタイトル
plt.xlabel("Time [s]")
plt.ylabel("Estimation Error [mmHg]")
plt.title("BP Estimation Error vs Time")

# x軸を等間隔カテゴリにする
plt.xticks(x, labels)


# y軸の範囲（最小5）
plt.ylim(5, max(sbp_error) + 1)

# 凡例とグリッド
plt.legend()

plt.show()
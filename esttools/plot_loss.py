from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. 文件路径
# =========================
xlsx_path = Path(r"D:\PycharmProjects\wangxiao_code\esttools\all_loss.xlsx")
out_dir = Path(r"D:\PycharmProjects\wangxiao_code\esttools\loss_plots")
out_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 读取 Excel
# =========================
df = pd.read_excel(xlsx_path)

print("Excel列名：")
print(df.columns.tolist())

# 你的真实列名
STEP_COL = "训练步数"
LOSS_COLS = [
    "语义先验主导组",
    "全要素标准化组",
    "全要素基线组",
    "空间一致性反证组",
    "光照先验主导组",
]

# 检查列是否存在
required_cols = [STEP_COL] + LOSS_COLS
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"Excel 缺少这些列: {missing_cols}")

# =========================
# 3. 数据清理
# =========================
data = df[required_cols].copy()

# 转为数值
for col in required_cols:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# 去空
data = data.dropna(subset=[STEP_COL])

# 按训练步数排序
data = data.sort_values(STEP_COL)

# 若训练步数重复，保留最后一次
data = data.groupby(STEP_COL, as_index=False).last()

# 平滑函数
def smooth_series(series, window=15):
    return series.rolling(window=window, center=True, min_periods=1).mean()

# =========================
# 4. Matplotlib 全局设置
# =========================
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["axes.unicode_minus"] = False

# 如果你电脑有微软雅黑，中文一般能正常显示
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]

def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# =========================
# 5. 图1：全要素标准化组 vs 全要素基线组
# =========================
plt.figure(figsize=(8.6, 5.2))

for col in ["全要素标准化组", "全要素基线组"]:

    plt.plot(
        data[STEP_COL],
        smooth_series(data[col], window=15),
        linewidth=2.2,
        label=f"{col}"
    )

plt.xlabel("训练步数")
plt.ylabel("LOSS损失")
plt.title("全要素标准化组与全要素基线组训练损失收敛曲线对比")
plt.grid(True, alpha=0.3)
plt.legend()
save_fig(out_dir / "01_全要素标准化组_vs_全要素基线组.png")

# =========================
# 6. 图2：前期局部放大（0-1000 step）
# =========================
plt.figure(figsize=(8.6, 5.2))

zoom_data = data[data[STEP_COL] <= 1000]

for col in ["全要素标准化组", "全要素基线组"]:

    plt.plot(
        zoom_data[STEP_COL],
        smooth_series(zoom_data[col], window=15),
        linewidth=2.2,
        label=f"{col}"
    )

plt.xlabel("训练步数")
plt.ylabel("LOSS损失")
plt.title("全要素标准化组与全要素基线组前期训练损失曲线对比（0–1000 step）")
plt.grid(True, alpha=0.3)
plt.legend()
save_fig(out_dir / "02_前期局部放大_0_1000step.png")

# =========================
# 7. 图3：5个主实验组总体对比
# =========================
plt.figure(figsize=(9.2, 5.8))

for col in LOSS_COLS:
    plt.plot(
        data[STEP_COL],
        smooth_series(data[col], window=15),
        linewidth=2.2,
        label=col
    )

plt.xlabel("训练步数")
plt.ylabel("LOSS损失")
plt.title("不同实验组训练损失收敛曲线对比")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)
save_fig(out_dir / "03_不同实验组训练损失收敛曲线对比.png")

# =========================
# 8. 图4：原始曲线单独版（适合附录）
# =========================
plt.figure(figsize=(9.2, 5.8))

for col in LOSS_COLS:
    plt.plot(
        data[STEP_COL],
        data[col],
        linewidth=1.5,
        label=col
    )

plt.xlabel("训练步数")
plt.ylabel("LOSS损失")
plt.title("不同实验组训练损失原始曲线对比")
plt.grid(True, alpha=0.3)
plt.legend(ncol=2)
save_fig(out_dir / "04_不同实验组训练损失原始曲线对比.png")

# =========================
# 9. 导出清洗后的数据
# =========================
data.to_csv(out_dir / "loss_cleaned.csv", index=False, encoding="utf-8-sig")

print("\n绘图完成，输出目录：")
print(out_dir.resolve())
for p in sorted(out_dir.glob("*")):
    print(" -", p.name)
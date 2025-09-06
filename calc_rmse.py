import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. 配置路径
# ---------------------------
excel_path = "/root/TCD_VOIP/MOS_LQS.xlsx"   # Excel 文件，Sheet 3
csv_root = "./visqol_csv"                    # ViSQOL 输出 CSV 根目录

# ---------------------------
# 2. 读取 MOS-LQS（主观评分）
# ---------------------------
df_mos = pd.read_excel(excel_path, sheet_name=2)  # Sheet 3
df_mos = df_mos[['Filename', 'ConditionID', 'sample MOS']]
df_mos.rename(columns={'sample MOS': 'MOS_LQS'}, inplace=True)
df_mos['Filename'] = df_mos['Filename'].str.strip()
df_mos['MOS_LQS'] = df_mos['MOS_LQS'].astype(float)

# ---------------------------
# 3. 递归读取 ViSQOL 输出 CSV
# ---------------------------
moslqo_list = []

for root, dirs, files in os.walk(csv_root):
    for file in files:
        if file.endswith(".csv"):
            path = os.path.join(root, file)
            df_csv = pd.read_csv(path)
            # MOS-LQO 在 CSV 第三列
            mos_lqo = float(df_csv.iloc[0, 2])
            # 文件名对齐 Excel (CSV C_01_ECHO_FA.csv -> Excel C_01_ECHO_FA.wav)
            filename = file.replace(".csv", ".wav")
            moslqo_list.append((filename, mos_lqo))

df_visqol = pd.DataFrame(moslqo_list, columns=['Filename', 'MOS_LQO'])
df_visqol['Filename'] = df_visqol['Filename'].str.strip()
df_visqol['MOS_LQO'] = df_visqol['MOS_LQO'].astype(float)

# ---------------------------
# 4. 合并 MOS-LQS 和 MOS-LQO
# ---------------------------
df = pd.merge(df_mos, df_visqol, on='Filename', how='inner')
print(f"合并后样本数: {len(df)}")
if df.empty:
    raise ValueError("合并后 DataFrame 为空，请检查文件名是否匹配")

# ---------------------------
# 5. 提取劣化类型
# ---------------------------
def extract_type(filename):
    fname_upper = filename.upper()
    if 'CHOP' in fname_upper:
        return 'chop'
    elif 'CLIP' in fname_upper:
        return 'clip'
    elif 'COMPSPKR' in fname_upper:
        return 'compspkr'
    elif 'ECHO' in fname_upper:
        return 'echo'
    elif 'NOISE' in fname_upper:
        return 'noise'
    else:
        return 'other'

df['Degradation'] = df['Filename'].apply(extract_type)

# ---------------------------
# 6. 计算整体RMSE
# ---------------------------
rmse_overall = np.sqrt(np.mean((df['MOS_LQS'] - df['MOS_LQO'])**2))
print(f"\n整体 RMSE: {rmse_overall:.4f}")

# ---------------------------
# 7. 按劣化类型分组RMSE
# ---------------------------
rmse_by_type = df.groupby('Degradation')[['MOS_LQS', 'MOS_LQO']].apply(
    lambda x: np.sqrt(np.mean((x['MOS_LQS'] - x['MOS_LQO'])**2))
)
print("\n按劣化类型分组 RMSE:")
print(rmse_by_type)

# ---------------------------
# 8. 保存结果
# ---------------------------
df.to_csv("merged_mos.csv", index=False)
rmse_by_type.to_csv("rmse_by_type.csv", header=True)
print("\n结果已保存：merged_mos.csv 和 rmse_by_type.csv")

# ---------------------------
# 9. 绘制分条件 RMSE 柱状图
# ---------------------------
plt.figure(figsize=(8,5))
rmse_by_type_sorted = rmse_by_type.sort_values()  # 可选：按 RMSE 排序
plt.bar(rmse_by_type_sorted.index, rmse_by_type_sorted.values, color='skyblue')
plt.xlabel("Degradation Type")
plt.ylabel("RMSE")
plt.title("RMSE by Degradation Type (ViSQOL)")
plt.ylim(0, max(rmse_by_type_sorted.values)*1.2)  # 适当留白
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加每个柱顶部数值
for i, v in enumerate(rmse_by_type_sorted.values):
    plt.text(i, v + 0.03, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("rmse_by_type.png", dpi=300)
plt.show()

# ---------------------------
# 10. 绘制总图和单独子图
# ---------------------------
deg_types = df['Degradation'].unique()
n_types = len(deg_types)

# 单独输出每组个体点图
for deg_type in deg_types:
    subset = df[df['Degradation']==deg_type]
    plt.figure(figsize=(12,4))
    plt.plot(subset['Filename'], subset['MOS_LQS'], color='blue', marker='o', linestyle='-', alpha=0.7, label='MOS-LQS')
    plt.plot(subset['Filename'], subset['MOS_LQO'], color='red', marker='x', linestyle='-', alpha=0.7, label='MOS-LQO')
    plt.xticks(rotation=90)
    plt.ylabel("MOS Value")
    plt.title(f"Individual MOS - {deg_type}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'individual_{deg_type}.png', dpi=300)
    plt.close()

# 单独输出组间平均点图
group_avg = df.groupby('Degradation')[['MOS_LQS','MOS_LQO']].mean()
plt.figure(figsize=(10,5))
plt.scatter(group_avg.index, group_avg['MOS_LQS'], color='blue', s=100, label='Average MOS-LQS')
plt.scatter(group_avg.index, group_avg['MOS_LQO'], color='red', s=100, label='Average MOS-LQO')
plt.plot(group_avg.index, group_avg['MOS_LQS'], color='blue', linestyle='--', alpha=0.7)
plt.plot(group_avg.index, group_avg['MOS_LQO'], color='red', linestyle='--', alpha=0.7)
plt.title("Average MOS per Degradation Type")
plt.ylabel("Average MOS")
plt.xlabel("Degradation Type")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("average_mos.png", dpi=300)
plt.show()

# 绘制总图：每组个体点 + 组间平均点
fig, axes = plt.subplots(n_types+1, 1, figsize=(15, 3*(n_types+1)), constrained_layout=True)

# 左侧：个体点状图（含折线）
for i, deg_type in enumerate(deg_types):
    subset = df[df['Degradation']==deg_type]
    axes[i].plot(subset['Filename'], subset['MOS_LQS'], color='blue', marker='o', linestyle='-', alpha=0.6, label='MOS-LQS')
    axes[i].plot(subset['Filename'], subset['MOS_LQO'], color='red', marker='x', linestyle='-', alpha=0.6, label='MOS-LQO')
    axes[i].set_title(f"Individual MOS - {deg_type}")
    axes[i].set_ylabel("MOS Value")
    axes[i].tick_params(axis='x', rotation=90)
    axes[i].grid(axis='y', linestyle='--', alpha=0.5)
    if i == 0:
        axes[i].legend()

# 右下：组间平均点 + 折线
group_avg = df.groupby('Degradation')[['MOS_LQS','MOS_LQO']].mean()
axes[-1].scatter(group_avg.index, group_avg['MOS_LQS'], color='blue', s=100, label='Average MOS-LQS')
axes[-1].scatter(group_avg.index, group_avg['MOS_LQO'], color='red', s=100, label='Average MOS-LQO')
axes[-1].plot(group_avg.index, group_avg['MOS_LQS'], color='blue', linestyle='--', alpha=0.7)
axes[-1].plot(group_avg.index, group_avg['MOS_LQO'], color='red', linestyle='--', alpha=0.7)
axes[-1].set_title("Average MOS per Degradation Type")
axes[-1].set_ylabel("Average MOS")
axes[-1].set_xlabel("Degradation Type")
axes[-1].legend()
axes[-1].grid(axis='y', linestyle='--', alpha=0.5)

plt.savefig("mos_total_plot.png", dpi=300)
plt.show()


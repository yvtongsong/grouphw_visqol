import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# ---------------------------
# 1. 配置路径
# ---------------------------
excel_path = "./MOS_LQS.xlsx"   # Excel 文件，Sheet 3
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
            mos_lqo = float(df_csv.iloc[0, 2])  # CSV 第三列
            filename = file.replace(".csv", ".wav")  # 对齐 Excel
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
# 6. 计算整体 RMSE
# ---------------------------
rmse_overall = np.sqrt(np.mean((df['MOS_LQS'] - df['MOS_LQO'])**2))
print(f"\n整体 RMSE: {rmse_overall:.4f}")

# ---------------------------
# 7. 按劣化类型分组 RMSE
# ---------------------------
rmse_by_type = df.groupby('Degradation')[['MOS_LQS', 'MOS_LQO']].apply(
    lambda x: np.sqrt(np.mean((x['MOS_LQS'] - x['MOS_LQO'])**2))
)
print("\n按劣化类型分组 RMSE:")
print(rmse_by_type)

# ---------------------------
# 8. 计算整体 PCC 和 SROCC
# ---------------------------
pcc_overall, _ = pearsonr(df['MOS_LQS'], df['MOS_LQO'])
srocc_overall, _ = spearmanr(df['MOS_LQS'], df['MOS_LQO'])
print(f"\n整体 PCC: {pcc_overall:.4f}")
print(f"整体 SROCC: {srocc_overall:.4f}")

# ---------------------------
# 9. 按劣化类型计算 PCC 和 SROCC
# ---------------------------
deg_types = df['Degradation'].unique()
corr_by_type = {}
for deg_type in deg_types:
    subset = df[df['Degradation'] == deg_type]
    pcc, _ = pearsonr(subset['MOS_LQS'], subset['MOS_LQO'])
    srocc, _ = spearmanr(subset['MOS_LQS'], subset['MOS_LQO'])
    corr_by_type[deg_type] = {'PCC': pcc, 'SROCC': srocc}

corr_df = pd.DataFrame(corr_by_type).T
print("\n按劣化类型 PCC 和 SROCC:")
print(corr_df)

# ---------------------------
# 10. 保存结果
# ---------------------------
df.to_csv("merged_mos.csv", index=False)
rmse_by_type.to_csv("rmse_by_type.csv", header=True)
corr_df.to_csv("corr_by_type.csv", index=True)
print("\n结果已保存：merged_mos.csv, rmse_by_type.csv, corr_by_type.csv")

# ---------------------------
# 11. 绘制 RMSE 柱状图
# ---------------------------
plt.figure(figsize=(8,5))
rmse_by_type_sorted = rmse_by_type.sort_values()
plt.bar(rmse_by_type_sorted.index, rmse_by_type_sorted.values, color='skyblue')
plt.xlabel("Degradation Type")
plt.ylabel("RMSE")
plt.title("RMSE by Degradation Type (ViSQOL)")
plt.ylim(0, max(rmse_by_type_sorted.values)*1.2)
for i, v in enumerate(rmse_by_type_sorted.values):
    plt.text(i, v + 0.03, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("rmse_by_type.png", dpi=300)
plt.show()

# ---------------------------
# 12. 单独个体点图 & 平均点图
# ---------------------------
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

# ---------------------------
# 13. 绘制 RMSE + PCC + SROCC 总图
# ---------------------------
deg_types_sorted = rmse_by_type.sort_values().index
rmse_vals = rmse_by_type[deg_types_sorted].values
pcc_vals = [corr_by_type[t]['PCC'] for t in deg_types_sorted]
srocc_vals = [corr_by_type[t]['SROCC'] for t in deg_types_sorted]

fig, ax1 = plt.subplots(figsize=(12,6))

# RMSE 柱状
ax1.bar(deg_types_sorted, rmse_vals, color='skyblue', label='RMSE')
ax1.set_ylabel('RMSE', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.set_ylim(0, max(rmse_vals)*1.2)
for i, v in enumerate(rmse_vals):
    ax1.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom', color='blue')

# PCC & SROCC 折线
ax2 = ax1.twinx()
ax2.plot(deg_types_sorted, pcc_vals, color='red', marker='o', linestyle='-', label='PCC')
ax2.plot(deg_types_sorted, srocc_vals, color='purple', marker='x', linestyle='--', label='SROCC')
ax2.set_ylabel('Correlation', color='black')
ax2.set_ylim(0, 1.05)
for i, (p, s) in enumerate(zip(pcc_vals, srocc_vals)):
    ax2.text(i, p + 0.02, f"{p:.2f}", ha='center', color='red')
    ax2.text(i, s + 0.02, f"{s:.2f}", ha='center', color='purple')

lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper right')

plt.title("RMSE + PCC + SROCC by Degradation Type")
plt.tight_layout()
plt.savefig("rmse_pcc_srocc.png", dpi=300)
plt.show()

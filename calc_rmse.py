import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1️⃣ 配置路径
# ---------------------------
excel_path = "/root/TCD_VOIP/MOS_LQS.xlsx"   # Excel 文件，Sheet 3
csv_root = "./visqol_csv"                    # ViSQOL 输出 CSV 根目录

# ---------------------------
# 2️⃣ 读取 MOS-LQS（主观评分）
# ---------------------------
df_mos = pd.read_excel(excel_path, sheet_name=2)  # Sheet 3
df_mos = df_mos[['Filename', 'ConditionID', 'sample MOS']]
df_mos.rename(columns={'sample MOS': 'MOS_LQS'}, inplace=True)
df_mos['Filename'] = df_mos['Filename'].str.strip()
df_mos['MOS_LQS'] = df_mos['MOS_LQS'].astype(float)

# ---------------------------
# 3️⃣ 递归读取 ViSQOL 输出 CSV
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
# 4️⃣ 合并 MOS-LQS 和 MOS-LQO
# ---------------------------
df = pd.merge(df_mos, df_visqol, on='Filename', how='inner')
print(f"合并后样本数: {len(df)}")
if df.empty:
    raise ValueError("合并后 DataFrame 为空，请检查文件名是否匹配")

# ---------------------------
# 5️⃣ 提取劣化类型
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
# 6️⃣ 计算整体 RMSE
# ---------------------------
rmse_overall = np.sqrt(np.mean((df['MOS_LQS'] - df['MOS_LQO'])**2))
print(f"\n整体 RMSE: {rmse_overall:.4f}")

# ---------------------------
# 7️⃣ 按劣化类型分组 RMSE
# ---------------------------
rmse_by_type = df.groupby('Degradation')[['MOS_LQS', 'MOS_LQO']].apply(
    lambda x: np.sqrt(np.mean((x['MOS_LQS'] - x['MOS_LQO'])**2))
)
print("\n按劣化类型分组 RMSE:")
print(rmse_by_type)

# ---------------------------
# 8️⃣ 保存结果
# ---------------------------
df.to_csv("merged_mos.csv", index=False)
rmse_by_type.to_csv("rmse_by_type.csv", header=True)
print("\n结果已保存：merged_mos.csv 和 rmse_by_type.csv")

# ---------------------------
# 9️⃣ 绘制分条件 RMSE 柱状图
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

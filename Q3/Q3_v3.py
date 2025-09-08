import pandas as pd
import numpy as np
import jenkspy
from lifelines import KaplanMeierFitter
from scipy.special import expit
import matplotlib.pyplot as plt
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

# 读取数据
df = pd.read_excel("initial_data/nipt_data.xlsx", sheet_name="男胎检测数据")
df = df[df['Y染色体浓度'].notna()].copy()

def convert_ga_to_decimal(ga_str, error_list):
    """将孕周字符串转换为小数形式"""
    if pd.isna(ga_str):
        error_list.append(ga_str)
        return np.nan
    ga_str = str(ga_str).lower().replace(' ', '')
    if ga_str.endswith('w') and ga_str[:-1].isdigit():
        return float(ga_str[:-1])
    for sep in ['w+', 'w', '+']:
        if sep in ga_str:
            parts = ga_str.split(sep)
            if len(parts) == 2:
                try:
                    return int(parts[0]) + int(parts[1]) / 7.0
                except ValueError:
                    error_list.append(ga_str)
                    return np.nan
    error_list.append(ga_str)
    return np.nan

# 数据预处理
error_list = []
df['小数孕周'] = df['检测孕周'].apply(lambda x: convert_ga_to_decimal(x, error_list))
df = df[(df['小数孕周'] >= 10) & (df['小数孕周'] <= 25)]
df['GC含量_百分比'] = df['GC含量'] * 100
df = df[(df['GC含量_百分比'] >= 40) & (df['GC含量_百分比'] <= 60)]
df['Y染色体浓度_百分比'] = df['Y染色体浓度'] * 100
df = df[(df['Y染色体浓度_百分比'] >= 0) & (df['Y染色体浓度_百分比'] <= 15)]

# 处理分类变量
df['IVF妊娠'] = df['IVF妊娠'].replace({
    '自然受孕': 0,
    'IUI（人工授精）': 1,
    'IVF（试管婴儿）': 2    
}).astype(int)

# 处理怀孕次数
def map_pregnancy_count(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, str):
        if x.startswith('≥'):
            return 3
        elif x.isdigit():
            return int(x)
    try:
        return int(x)
    except:
        return np.nan

df["怀孕次数_类型"] = df["怀孕次数"].apply(map_pregnancy_count).astype('Int64')

# 核心数据数组
Y_data_whole = df['Y染色体浓度'].values
wk_data_whole = df['小数孕周'].values
BMI_data_whole = df['孕妇BMI'].values
IVF_data_whole = df['IVF妊娠'].values
Pregnancy_count_data_whole = df['怀孕次数_类型'].fillna(1).values  # 填充缺失值
age_data_whole = df['年龄'].values
height_data_whole = df['身高'].values
weight_data_whole = df['体重'].values

# 事件定义：Y染色体浓度 ≥ 0.04 为未达标
event_observed = (Y_data_whole >= 0.04).astype(int)

print(f"数据样本数: {len(BMI_data_whole)}")
print(f"BMI范围: {np.min(BMI_data_whole):.2f} - {np.max(BMI_data_whole):.2f}")
print(f"未达标样本数: {np.sum(event_observed)}")
print(f"未达标比例: {np.mean(event_observed):.4f}")

# -----------------------
# 风险函数定义
# -----------------------
def risk_time(time):
    """孕周风险：孕周越接近21周风险越低"""
    return expit(-0.1 * (time - 21))

def risk_HIV(IVF_data):
    """IVF风险：IVF妊娠风险较高"""
    if IVF_data < 0.5:
        return 0.4
    else:
        return 0.8


def risk_age(age_data):
    """年龄风险：年龄越接近30岁风险越低"""
    return expit(-0.1 * (age_data - 30))

def risk_count(Pregnancy_count_data):
    """怀孕次数风险：怀孕次数越多风险越高"""
    if Pregnancy_count_data < 0.5:
        return 0.2
    elif Pregnancy_count_data < 1.5:
        return 0.3
    else:
        return 0.8

def risk_height(height_data):
    """身高风险：身高越接近160cm风险越低"""
    return expit(-0.1 * (height_data - 160))

def risk_heavy(weight_data):
    """体重风险：体重越接近60kg风险越低"""
    return expit(-0.1 * (weight_data - 60))

def get_survival_probability(kmf_dict, label, time_point):
    """
    从kmf获取生存率，若无该组或无法预测则返回1.0
    """
    if label not in kmf_dict:
        return 1.0
    kmf = kmf_dict[label]
    try:
        sf = kmf.survival_function_at_times(time_point)
        if hasattr(sf, 'values'):
            val = float(np.array(sf).ravel()[0])
        else:
            val = float(sf)
        return val if not np.isnan(val) else 1.0
    except Exception:
        return 1.0

def get_failure_probability(kmf_dict, label, time_point):
    return 1.0 - get_survival_probability(kmf_dict, label, time_point)

def comprehensive_loss(time, IVF_data, age_data, Pregnancy_count_data, 
                      height_data, weight_data, label, kmf_dict, alpha=1.0, beta=1.0):
    """
    计算单个样本的综合损失
    """
    p_t = get_failure_probability(kmf_dict, label, time)
    risk = (
        risk_time(time) +
        risk_HIV(IVF_data) +
        risk_age(age_data) +
        risk_count(Pregnancy_count_data) +
        risk_height(height_data) +
        risk_heavy(weight_data)
    )
    total_risk = alpha * (1 - p_t) + beta * risk
    return total_risk, p_t

# -----------------------
# 优化算法
# -----------------------
def find_optimal_bmi_bins(BMI_data, n_groups=3, n_iter=20):
    """
    寻找最优BMI区间划分
    """
    best_loss = float('inf')
    best_bins = None
    best_kmf_dict = None
    
    min_bmi, max_bmi = np.min(BMI_data), np.max(BMI_data)
    
    print(f"BMI范围: {min_bmi:.2f} - {max_bmi:.2f}")
    
    # 方法1：使用分位数作为初始分割点
    quantiles = np.linspace(0, 1, n_groups + 1)[1:-1]
    quantile_bins = np.quantile(BMI_data, quantiles)
    
    # 方法2：等间距分割
    equal_bins = np.linspace(min_bmi, max_bmi, n_groups + 1)[1:-1]
    
    candidate_sets = [quantile_bins, equal_bins]
    
    # 尝试不同的初始分割方案
    for candidate_idx, candidate_bins in enumerate(candidate_sets):
        print(f"尝试候选方案 {candidate_idx + 1}...")
        
        bins = np.concatenate([[min_bmi], candidate_bins, [max_bmi]])
        bins = np.sort(bins)
        
        labels = np.digitize(BMI_data, bins) - 1
        unique_labels = np.unique(labels)
        
        if len(unique_labels) < n_groups:
            continue
        
        # 计算KM曲线
        kmf_dict = {}
        for label in unique_labels:
            mask = (labels == label)
            if np.sum(mask) > 5:  # 确保有足够样本
                kmf = KaplanMeierFitter()
                kmf.fit(wk_data_whole[mask], event_observed=event_observed[mask])
                kmf_dict[label] = kmf
        
        if len(kmf_dict) < n_groups:
            continue
        
        # 计算总损失
        total_loss = 0
        for j in range(len(BMI_data)):
            loss_j, _ = comprehensive_loss(
                wk_data_whole[j], IVF_data_whole[j], age_data_whole[j],
                Pregnancy_count_data_whole[j], height_data_whole[j],
                weight_data_whole[j], labels[j], kmf_dict
            )
            total_loss += loss_j
        
        if total_loss < best_loss:
            best_loss = total_loss
            best_bins = bins
            best_kmf_dict = kmf_dict.copy()
            print(f"  当前最佳损失: {best_loss:.4f}")
    
    # 局部优化
    if best_bins is not None:
        print("进行局部优化...")
        for i in range(n_iter):
            # 在最佳分割点附近进行扰动
            perturbation = np.random.normal(0, 0.5, len(best_bins)-2)  # 减小扰动幅度
            new_bins = best_bins[1:-1] + perturbation
            new_bins = np.clip(new_bins, min_bmi + 0.1, max_bmi - 0.1)  # 避免边界问题
            bins = np.concatenate([[min_bmi], np.sort(new_bins), [max_bmi]])
            
            labels = np.digitize(BMI_data, bins) - 1
            unique_labels = np.unique(labels)
            
            if len(unique_labels) < n_groups:
                continue
            
            kmf_dict = {}
            valid_groups = 0
            for label in unique_labels:
                mask = (labels == label)
                if np.sum(mask) > 5:
                    kmf = KaplanMeierFitter()
                    kmf.fit(wk_data_whole[mask], event_observed=event_observed[mask])
                    kmf_dict[label] = kmf
                    valid_groups += 1
            
            if valid_groups < n_groups:
                continue
            
            total_loss = 0
            for j in range(len(BMI_data)):
                loss_j, _ = comprehensive_loss(
                    wk_data_whole[j], IVF_data_whole[j], age_data_whole[j],
                    Pregnancy_count_data_whole[j], height_data_whole[j],
                    weight_data_whole[j], labels[j], kmf_dict
                )
                total_loss += loss_j
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_bins = bins
                best_kmf_dict = kmf_dict.copy()
                print(f"  迭代 {i+1}: 损失改善至 {best_loss:.4f}")
    
    return best_bins, best_kmf_dict, best_loss

# -----------------------
# 主程序
# -----------------------
# 设置要划分的组数
n_groups = 3

print(f"开始寻找最优BMI区间划分，目标组数: {n_groups}...")
start_time = time.time()

optimal_bins, optimal_kmf_dict, optimal_loss = find_optimal_bmi_bins(
    BMI_data_whole, n_groups=n_groups, n_iter=10
)

end_time = time.time()
print(f"优化完成，耗时: {end_time - start_time:.2f}秒")

# 如果优化失败，使用等分区间
if optimal_bins is None:
    print("使用等分区间作为备选方案")
    optimal_bins = np.linspace(np.min(BMI_data_whole), np.max(BMI_data_whole), n_groups + 1)
    labels = np.digitize(BMI_data_whole, optimal_bins) - 1
    
    optimal_kmf_dict = {}
    for label in np.unique(labels):
        mask = (labels == label)
        if np.sum(mask) > 0:
            kmf = KaplanMeierFitter()
            kmf.fit(wk_data_whole[mask], event_observed=event_observed[mask])
            optimal_kmf_dict[label] = kmf
    
    optimal_loss = 0
    for i in range(len(BMI_data_whole)):
        loss_i, _ = comprehensive_loss(
            wk_data_whole[i], IVF_data_whole[i], age_data_whole[i],
            Pregnancy_count_data_whole[i], height_data_whole[i],
            weight_data_whole[i], labels[i], optimal_kmf_dict
        )
        optimal_loss += loss_i

print(f"\n最优分割点: {[f'{x:.2f}' for x in optimal_bins]}")
print(f"最小损失值: {optimal_loss:.4f}")

# 使用最优分割点划分数据
labels = np.digitize(BMI_data_whole, optimal_bins) - 1

# 计算每个区间的统计信息
print("\n各区间详细统计信息:")
for i in range(n_groups):
    mask = (labels == i)
    n_samples = np.sum(mask)
    bmi_min = optimal_bins[i]
    bmi_max = optimal_bins[i+1]
    
    if n_samples > 0:
        median_week = np.median(wk_data_whole[mask])
        failure_rate = 1 - get_survival_probability(optimal_kmf_dict, i, median_week)
        
        print(f"区间 {i} (BMI: {bmi_min:.1f}-{bmi_max:.1f})")
        print(f"  样本数: {n_samples} ({n_samples/len(BMI_data_whole)*100:.1f}%)")
        print(f"  未达标率: {failure_rate:.4f}")
        print(f"  BMI: {np.mean(BMI_data_whole[mask]):.1f} ± {np.std(BMI_data_whole[mask]):.1f}")
        print(f"  Y浓度: {np.mean(Y_data_whole[mask]):.6f} ± {np.std(Y_data_whole[mask]):.6f}")
        print(f"  孕周: {np.mean(wk_data_whole[mask]):.1f} ± {np.std(wk_data_whole[mask]):.1f}")
        print(f"  年龄: {np.mean(age_data_whole[mask]):.1f} ± {np.std(age_data_whole[mask]):.1f}")
        print(f"  怀孕次数: {np.mean(Pregnancy_count_data_whole[mask]):.1f} ± {np.std(Pregnancy_count_data_whole[mask]):.1f}")
    else:
        print(f"区间 {i} (BMI: {bmi_min:.1f}-{bmi_max:.1f}) 无样本")
    print("-" * 60)

# 可视化结果
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. KM生存曲线
for i in range(n_groups):
    if i in optimal_kmf_dict:
        optimal_kmf_dict[i].plot_survival_function(
            ax=axes[0, 0], 
            label=f'区间{i} (BMI: {optimal_bins[i]:.1f}-{optimal_bins[i+1]:.1f})'
        )
axes[0, 0].set_title('各BMI区间KM生存曲线')
axes[0, 0].set_xlabel('孕周')
axes[0, 0].set_ylabel('生存概率（达标率）')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. BMI分布和分割点
axes[0, 1].hist(BMI_data_whole, bins=30, alpha=0.7, density=True, color='skyblue')
for bin_edge in optimal_bins:
    axes[0, 1].axvline(x=bin_edge, color='red', linestyle='--', alpha=0.8, linewidth=2)
axes[0, 1].set_title('BMI分布及最优分割点')
axes[0, 1].set_xlabel('BMI')
axes[0, 1].set_ylabel('密度')
axes[0, 1].grid(True, alpha=0.3)

# 3. 各区间未达标率
failure_rates = []
bin_ranges = []
for i in range(n_groups):
    if i in optimal_kmf_dict:
        mask = (labels == i)
        median_week = np.median(wk_data_whole[mask])
        failure_rate = 1 - get_survival_probability(optimal_kmf_dict, i, median_week)
        failure_rates.append(failure_rate)
        bin_ranges.append(f"{optimal_bins[i]:.1f}-{optimal_bins[i+1]:.1f}")

axes[1, 0].bar(bin_ranges, failure_rates, alpha=0.7, color='lightcoral')
axes[1, 0].set_title('各BMI区间未达标率')
axes[1, 0].set_xlabel('BMI区间')
axes[1, 0].set_ylabel('未达标率')
axes[1, 0].grid(True, alpha=0.3)

# 4. 各区间Y染色体浓度分布
for i in range(n_groups):
    mask = (labels == i)
    if np.sum(mask) > 0:
        axes[1, 1].hist(Y_data_whole[mask], bins=20, alpha=0.6, 
                       label=f'区间{i}', density=True)
axes[1, 1].axvline(x=0.04, color='red', linestyle='--', alpha=0.8, label='达标阈值(0.04)')
axes[1, 1].set_title('各BMI区间的Y染色体浓度分布')
axes[1, 1].set_xlabel('Y染色体浓度')
axes[1, 1].set_ylabel('密度')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'optimal_bmi_analysis_{n_groups}_groups.png', dpi=300, bbox_inches='tight')
plt.show()

# 保存详细结果
results_df = df.copy()
results_df['BMI区间标签'] = labels
results_df['BMI区间下限'] = [optimal_bins[label] for label in labels]
results_df['BMI区间上限'] = [optimal_bins[label] for label in labels]

# 计算每个样本的损失值和未达标概率
loss_values = []
failure_probs = []
for i in range(len(BMI_data_whole)):
    loss_i, p_t_i = comprehensive_loss(
        wk_data_whole[i], IVF_data_whole[i], age_data_whole[i],
        Pregnancy_count_data_whole[i], height_data_whole[i],
        weight_data_whole[i], labels[i], optimal_kmf_dict
    )
    loss_values.append(loss_i)
    failure_probs.append(p_t_i)

results_df['综合损失值'] = loss_values
results_df['未达标概率'] = failure_probs

# 保存结果
output_filename = f'optimal_bmi_binning_results_{n_groups}_groups.xlsx'
results_df.to_excel(output_filename, index=False)
print(f"\n分析完成！结果已保存到 {output_filename}")
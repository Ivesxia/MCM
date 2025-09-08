import pandas as pd
import numpy as np
import jenkspy
from lifelines import KaplanMeierFitter
from scipy.special import expit
from itertools import combinations
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False 

df = pd.read_excel("initial_data/nipt_data.xlsx", sheet_name="男胎检测数据")
df = df[df['Y染色体浓度'].notna()].copy()

def convert_ga_to_decimal(ga_str, error_list):
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

error_list = []
df['小数孕周'] = df['检测孕周'].apply(lambda x: convert_ga_to_decimal(x, error_list))
df = df[(df['小数孕周'] >= 10) & (df['小数孕周'] <= 25)]
df['GC含量_百分比'] = df['GC含量'] * 100
df = df[(df['GC含量_百分比'] >= 40) & (df['GC含量_百分比'] <= 60)]
df['Y染色体浓度_百分比'] = df['Y染色体浓度'] * 100
df = df[(df['Y染色体浓度_百分比'] >= 0) & (df['Y染色体浓度_百分比'] <= 15)]

df['IVF妊娠'] = df['IVF妊娠'].replace({
    '自然受孕': 0,
    'IUI（人工授精）': 1,
    'IVF（试管婴儿）': 2    
})

category_mapping = {"1": 1, "2": 2, "≥3": 3}
df["怀孕次数_类型"] = df["怀孕次数"].map(category_mapping)

# 核心数组
Y_data_whole = df['Y染色体浓度'].values
wk_data_whole = df['小数孕周'].values
BMI_data_whole = df['孕妇BMI'].values
IVF_data_whole = df['IVF妊娠'].values
Pregnancy_count_data_whole = df['怀孕次数_类型'].values
age_data_whole = df['年龄'].values
height_data_whole = df['身高'].values
weight_data_whole = df['体重'].values
event_observed = (Y_data_whole >= 0.04).astype(int)

print(f"数据样本数: {len(Y_data_whole)}")

# -----------------------
# 风险函数：保持你的定义
# -----------------------
def risk_time(time):
    return expit(-0.1 * (time - 21))

def risk_HIV(IVF_data):
    return expit(-0.1 * (IVF_data - 1))

def risk_age(age_data):
    return expit(-0.1 * (age_data - 30))

def risk_count(Pregnancy_count_data):
    return expit(-0.1 * (Pregnancy_count_data - 1))

def risk_height(height_data):
    return expit(-0.1 * (height_data - 160))

def risk_heavy(weight_data):
    return expit(-0.1 * (weight_data - 60))

# -----------------------
# 分箱与KM拟合等函数（修正）
# -----------------------
def create_labels_from_breaks(bmi_values, breaks):
    """
    返回 0..(len(breaks)-2) 的标签。breaks 长度为 n_groups+1，表示闭区间边界 [min, ..., max]
    使用 np.digitize，把区间划分为：[breaks[0], breaks[1]), [breaks[1], breaks[2]), ..., [breaks[-2], breaks[-1]]
    """
    # bins 为内部切点（不含第一个和最后一个）
    bins = np.array(breaks[1:-1]) if len(breaks) > 2 else np.array([])
    labels = np.digitize(bmi_values, bins=bins, right=False)  # labels in 0..len(breaks)-2
    # 确保范围
    labels = np.clip(labels, 0, len(breaks)-2)
    return labels

def fit_kaplan_meier(event_times, labels, event_observed, min_samples=6):
    kmf_dict = {}
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = (labels == label)
        if np.sum(mask) >= min_samples:
            kmf = KaplanMeierFitter()
            # lifelines 的 fit 接受 durations, event_observed
            kmf.fit(event_times[mask], event_observed=event_observed[mask])
            kmf_dict[int(label)] = kmf
    return kmf_dict

def get_survival_probability(kmf_dict, label, time_point):
    """
    更稳健地从 kmf 获取生存率，若无该组或无法预测则返回 1.0（表示生存率 100%）
    """
    if int(label) not in kmf_dict:
        return 1.0
    kmf = kmf_dict[int(label)]
    try:
        # 使用 survival_function_at_times，确保取到标量
        sf = kmf.survival_function_at_times(time_point)
        # 返回标量；如果是 Series，取第一个值
        if hasattr(sf, 'values'):
            val = float(np.array(sf).ravel()[0])
        else:
            val = float(sf)
        if np.isnan(val):
            return 1.0
        return val
    except Exception:
        return 1.0

def get_failure_probability(kmf_dict, label, time_point):
    return 1.0 - get_survival_probability(kmf_dict, label, time_point)

def comprehensive_loss(time, IVF_data, age_data, Pregnancy_count_data, height_data, weight_data, label, kmf_dict, alpha=1.0, beta=1.0):
    """
    返回 (total_risk_for_sample, p_t_of_group)
    注意：这里返回的是单个样本的损失（便于向量化）
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
# 优化函数（向量化改进）
# -----------------------
def optimize_group_times(kmf_dict, n_groups, labels, IVF_data, age_data, Pregnancy_count_data, height_data, weight_data, alpha=1.0, beta=1.0):
    optimal_times = [None] * n_groups
    optimal_losses = [np.inf] * n_groups

    for label in range(n_groups):
        if label not in kmf_dict:
            # 如果没有KM模型，保守地设置为中期20周并损失为 inf（或特定值）
            optimal_times[label] = 20.0
            optimal_losses[label] = float('inf')
            continue

        mask = (labels == label)
        if np.sum(mask) < 3:
            optimal_times[label] = 20.0
            optimal_losses[label] = float('inf')
            continue

        # 向量化：先提取组内样本的协变量
        IVF_grp = IVF_data[mask]
        age_grp = age_data[mask]
        count_grp = Pregnancy_count_data[mask]
        height_grp = height_data[mask]
        weight_grp = weight_data[mask]

        weeks = np.linspace(10, 25, 151)
        losses = np.zeros_like(weeks)

        for i, w in enumerate(weeks):
            # 组的达标概率（对所有样本相同）
            p_t = get_failure_probability(kmf_dict, label, w)
            # 组内每个样本的 risk 向量
            risk_vec = (risk_time(w) + risk_HIV(IVF_grp) + risk_age(age_grp) +
                        risk_count(count_grp) + risk_height(height_grp) + risk_heavy(weight_grp))
            # 每个样本的总损失 = alpha*(1-p_t) + beta * risk_vec
            sample_losses = alpha * (1.0 - p_t) + beta * risk_vec
            # 聚合为平均损失（也可以用 sum）
            losses[i] = np.mean(sample_losses)

        best_idx = np.argmin(losses)
        optimal_times[label] = float(weeks[best_idx])
        optimal_losses[label] = float(losses[best_idx])

    return optimal_times, optimal_losses

def calculate_total_loss(breaks, bmi_values, event_times, event_observed, IVF_data, age_data, Pregnancy_count_data, height_data, weight_data):
    labels = create_labels_from_breaks(bmi_values, breaks)
    group_sizes = [np.sum(labels == i) for i in range(len(breaks) - 1)]
    # 如果某组样本过少，返回 inf（不可接受）
    if any(size < 2 for size in group_sizes):
        return float('inf'), None, None, labels

    kmf_dict = fit_kaplan_meier(event_times, labels, event_observed, min_samples=6)
    n_groups = len(breaks) - 1
    optimal_times, optimal_losses = optimize_group_times(
        kmf_dict, n_groups, labels, IVF_data, age_data, Pregnancy_count_data, height_data, weight_data
    )

    total_loss = sum([l if l is not None else float('inf') for l in optimal_losses])
    return total_loss, optimal_times, kmf_dict, labels

# -----------------------
# 网格搜索（小心组合爆炸）
# -----------------------
def grid_search_optimization(n_groups=2, grid_points=8):
    min_bmi, max_bmi = float(np.min(BMI_data_whole)), float(np.max(BMI_data_whole))
    best_loss = float('inf')
    best_breaks = None
    best_times = None
    best_kmf = None
    best_labels = None

    print("正在进行网格搜索优化...（注意：组合数可能很大）")
    grid = np.linspace(min_bmi + 0.5, max_bmi - 0.5, grid_points)

    optimization_history = []
    # combinations(grid, n_groups-1) 可能会很大，必要时改用随机采样或减少 grid_points
    for inner_breaks in combinations(grid, n_groups - 1):
        breaks = sorted([min_bmi] + list(inner_breaks) + [max_bmi])
        try:
            total_loss, optimal_times, kmf_dict, labels = calculate_total_loss(
                breaks, BMI_data_whole, wk_data_whole, event_observed,
                IVF_data_whole, age_data_whole, Pregnancy_count_data_whole, height_data_whole, weight_data_whole
            )

            optimization_history.append({
                'breaks': breaks.copy(),
                'loss': total_loss,
                'times': optimal_times.copy() if optimal_times else None
            })

            if total_loss < best_loss:
                best_loss = total_loss
                best_breaks = breaks
                best_times = optimal_times
                best_kmf = kmf_dict
                best_labels = labels

        except Exception as e:
            # 出错时跳过该组合
            print(f"Error in processing combination {breaks}: {str(e)}")
            continue

    return best_breaks, best_times, best_kmf, best_loss, best_labels, optimization_history

# -----------------------
# 执行优化
# -----------------------
n_groups = 2
grid_points = 8

best_breaks, best_times, best_kmf, best_loss, best_labels, optimization_history = grid_search_optimization(
    n_groups=n_groups, grid_points=grid_points
)

# 计算Jenks分组结果（传入完整参数）
jenks_breaks = jenkspy.jenks_breaks(BMI_data_whole, n_groups)
jenks_loss, jenks_times, jenks_kmf, jenks_labels = calculate_total_loss(
    jenks_breaks, BMI_data_whole, wk_data_whole, event_observed,
    IVF_data_whole, age_data_whole, Pregnancy_count_data_whole, height_data_whole, weight_data_whole
)

# -----------------------
# 结果输出（修正打印，避免不匹配的解包）
# -----------------------
print("\n" + "=" * 60)
print("最终优化结果")
print("=" * 60)
print(f"数据统计:")
print(f"   总样本数: {len(BMI_data_whole)}")
print(f"   BMI范围: [{np.min(BMI_data_whole):.1f}, {np.max(BMI_data_whole):.1f}]")
print(f"   BMI均值: {np.mean(BMI_data_whole):.1f} ± {np.std(BMI_data_whole):.1f}")

if best_breaks is None:
    print("未找到可行的分组方案（可能数据量太少或网格设置问题）")
else:
    print(f"\n最优分组方案:")
    print(f"   BMI分组边界: {[f'{x:.1f}' for x in best_breaks]}")
    print(f"   总体损失: {best_loss:.6f}")

    print(f"\n 各组最优NIPT检测时间:")
    for i in range(len(best_breaks) - 1):
        group_size = int(np.sum(best_labels == i)) if best_labels is not None else 0
        if group_size > 0 and best_kmf is not None and i in best_kmf and best_times is not None:
            # 取该组的最优周
            t_opt = best_times[i]
            # 组的达标概率 p_t
            p_t = get_failure_probability(best_kmf, i, t_opt)
            # 组内平均风险（用现有样本向量化计算）
            mask = (best_labels == i)
            avg_risk = np.mean(
                risk_time(t_opt) + risk_HIV(IVF_data_whole[mask]) + risk_age(age_data_whole[mask]) +
                risk_count(Pregnancy_count_data_whole[mask]) + risk_height(height_data_whole[mask]) +
                risk_heavy(weight_data_whole[mask])
            )
            # 组内平均总损失（alpha=1,beta=1）
            avg_total_loss = np.mean(1.0 - p_t + (
                risk_time(t_opt) + risk_HIV(IVF_data_whole[mask]) + risk_age(age_data_whole[mask]) +
                risk_count(Pregnancy_count_data_whole[mask]) + risk_height(height_data_whole[mask]) +
                risk_heavy(weight_data_whole[mask])
            ))

            print(f"   组{i} (BMI {best_breaks[i]:.1f}-{best_breaks[i+1]:.1f}):")
            print(f"     → 样本数: {group_size}")
            print(f"     → 最优时间: {t_opt:.1f}周")
            print(f"     → 达标概率（组内）: {p_t:.3f}")
            print(f"     → 组内平均风险: {avg_risk:.4f}")
            print(f"     → 组内平均总损失: {avg_total_loss:.4f}")
        else:
            print(f"   组{i} (BMI {best_breaks[i]:.1f}-{best_breaks[i+1]:.1f}): 无足够样本或未拟合KM模型")

    print(f"\n 优化效果对比:")
    print(f"   Jenks分组损失: {jenks_loss:.6f}")
    print(f"   最优分组损失: {best_loss:.6f}")
    if jenks_loss != 0 and np.isfinite(jenks_loss):
        print(f"   优化提升: {(jenks_loss - best_loss)/jenks_loss*100:.1f}%")
    else:
        print("   无法计算优化提升（Jenks分组损失为 0 或无效）")

# 可视化部分：保留，但加了异常保护
try:
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    losses = [item['loss'] for item in optimization_history if np.isfinite(item['loss'])]
    iterations = range(len(losses))
    plt.plot(iterations, losses, '-', alpha=0.7, label='搜索过程中的损失')
    if np.isfinite(best_loss):
        plt.axhline(y=best_loss, color='r', linestyle='--', label=f'最优损失: {best_loss:.4f}')
    if np.isfinite(jenks_loss):
        plt.axhline(y=jenks_loss, color='g', linestyle='--', label=f'Jenks损失: {jenks_loss:.4f}')
    plt.xlabel('搜索迭代次数'); plt.ylabel('总体损失'); plt.title('网格搜索优化过程'); plt.legend(); plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(BMI_data_whole, bins=30, alpha=0.6, edgecolor='black', density=True)
    if best_breaks is not None:
        colors = ['red', 'blue', 'green', 'purple', 'cyan']
        for i in range(len(best_breaks)-1):
            plt.axvspan(best_breaks[i], best_breaks[i+1], alpha=0.15, color=colors[i % len(colors)])
        for break_point in best_breaks:
            plt.axvline(x=break_point, color='red', linestyle='-', linewidth=2, alpha=0.8)
    if jenks_breaks is not None:
        for break_point in jenks_breaks:
            plt.axvline(x=break_point, color='green', linestyle='--', linewidth=2, alpha=0.8)
    plt.xlabel('BMI'); plt.ylabel('密度'); plt.title('最优分组(实线) vs Jenks分组(虚线)'); plt.legend(['BMI分布', '最优分组', 'Jenks分组'])
    plt.tight_layout()
    plt.savefig('optimization_results_fixed.png', dpi=300, bbox_inches='tight')
    plt.show()
except Exception as e:
    print("绘图出错:", e)

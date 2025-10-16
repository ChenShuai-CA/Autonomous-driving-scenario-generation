import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 移除了sklearn依赖
import os

# 设置中文字体（如果可用）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def smooth_curve(values, window_size=10):
    """对曲线进行移动平均平滑处理"""
    if len(values) < window_size:
        return values
    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    # 为了保持原始长度，在开头添加原始值
    return np.concatenate([values[:window_size-1], smoothed])

def fit_trend_line(x, y, degree=2):
    """使用多项式拟合趋势线"""
    # 移除NaN值
    mask = ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < degree + 1:  # 需要足够的点进行拟合
        return x, np.full_like(x, np.nan)
    
    # 多项式拟合
    try:
        coeffs = np.polyfit(x_clean, y_clean, degree)
        y_pred = np.polyval(coeffs, x)
        return x, y_pred
    except Exception:
        # 如果拟合失败，使用线性拟合
        if len(x_clean) >= 2:
            coeffs = np.polyfit(x_clean, y_clean, 1)
            y_pred = np.polyval(coeffs, x)
            return x, y_pred
        else:
            return x, np.full_like(x, np.nan)

def plot_metric_curves(data, metric_name, y_label, output_dir, include_smooth_trend=True, title_suffix="变化曲线"):
    """绘制单个指标的曲线"""
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 提取数据
    steps = data['epoch'].values
    raw_values = data[metric_name].values
    
    # 绘制原始曲线（使用黑色，加粗线条）
    ax.plot(steps, raw_values, alpha=0.3, color='black', linewidth=1.5, label='原始值')
    
    # 如果需要平滑和趋势线
    if include_smooth_trend:
        # 平滑曲线
        smoothed_values = smooth_curve(raw_values, window_size=5)
        
        # 趋势线
        trend_x, trend_y = fit_trend_line(steps, raw_values, degree=3)
        
        # 绘制平滑曲线和趋势线
        ax.plot(steps, smoothed_values, color='blue', linewidth=2, label='平滑曲线')
        ax.plot(trend_x, trend_y, color='red', linewidth=2, linestyle='--', label='趋势线')
    
    # 设置图形属性
    ax.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} {title_suffix}', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 保存图形
    output_path = os.path.join(output_dir, f'{metric_name}_curve.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存 {metric_name} 曲线图到: {output_path}")

def plot_ade_fde_curves_with_min(data, metric_name, y_label, output_dir):
    """绘制ADE和FDE曲线（不包含平滑和趋势线，但标注最低值）"""
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 提取数据
    steps = data['epoch'].values
    raw_values = data[metric_name].values
    
    # 绘制原始曲线（使用黑色，加粗线条）
    ax.plot(steps, raw_values, alpha=0.3, color='black', linewidth=1.5, label='原始值')
    
    # 找到最低值及其位置
    min_idx = np.nanargmin(raw_values)
    min_step = steps[min_idx]
    min_value = raw_values[min_idx]
    
    # 在图上标注最低值点
    ax.scatter(min_step, min_value, color='red', s=100, zorder=5, label=f'最低值: {min_value:.4f}')
    ax.annotate(f'最低值: {min_value:.4f}\nEpoch: {min_step}', 
                xy=(min_step, min_value), 
                xytext=(min_step, min_value + (np.nanmax(raw_values) - np.nanmin(raw_values)) * 0.1),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10,
                ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    
    # 设置图形属性
    ax.set_xlabel('训练轮数 (Epoch)', fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'{y_label} 变化曲线', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 保存图形（使用curve.png而不是convergence.png）
    output_path = os.path.join(output_dir, f'{metric_name}_curve.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存 {metric_name} 变化曲线到: {output_path} (最低值: {min_value:.4f} at epoch {min_step})")

def main():
    # 创建输出目录
    output_dir = "training_curves"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取summary文件
    summary_path = "log_formal_mamba_component_weights/summary_10ep.txt"
    if os.path.exists(summary_path):
        # 读取summary文件
        summary_data = []
        with open(summary_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # 解析每一行
                    parts = line.strip().split('\t')
                    row = {}
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            try:
                                row[key] = float(value)
                            except ValueError:
                                row[key] = value
                    summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # 绘制总损失曲线
        if 'loss' in summary_df.columns:
            plot_metric_curves(summary_df, 'loss', '模型总损失 (Loss_total)', output_dir, title_suffix="收敛曲线")
        
        # 绘制风险分数曲线
        if 'risk_score' in summary_df.columns:
            plot_metric_curves(summary_df, 'risk_score', '风险分数 (Risk Score)', output_dir, title_suffix="变化曲线")
        
        # 绘制组件最大权重曲线
        if 'compw_max' in summary_df.columns:
            plot_metric_curves(summary_df, 'compw_max', '组件最大权重 (Component Max Weight)', output_dir, title_suffix="变化曲线")
        
        # 绘制组件熵曲线
        if 'compw_entropy' in summary_df.columns:
            plot_metric_curves(summary_df, 'compw_entropy', '组件熵 (Component Entropy)', output_dir, title_suffix="变化曲线")
    
    # 读取eval_log文件
    eval_log_path = "log_formal_mamba_component_weights/eval_log.csv"
    if os.path.exists(eval_log_path):
        eval_df = pd.read_csv(eval_log_path)
        
        # 绘制ADE曲线（变化曲线）
        if 'ade' in eval_df.columns:
            plot_ade_fde_curves_with_min(eval_df, 'ade', '平均位移误差 (ADE)', output_dir)
        
        # 绘制FDE曲线（变化曲线）
        if 'fde' in eval_df.columns:
            plot_ade_fde_curves_with_min(eval_df, 'fde', '最终位移误差 (FDE)', output_dir)
    
    # 读取autoscale_log文件
    autoscale_path = "log_formal_mamba_component_weights/autoscale_log.csv"
    if os.path.exists(autoscale_path):
        autoscale_df = pd.read_csv(autoscale_path)
        
        # 绘制风险全局缩放因子曲线
        if 'risk_global_scale' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'risk_global_scale', '风险全局缩放因子 (Risk Global Scale)', output_dir, title_suffix="变化曲线")
        
        # 绘制风险带损失曲线
        if 'risk_L_band' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'risk_L_band', '风险带损失 (Risk Band Loss)', output_dir, title_suffix="变化曲线")
        
        # 绘制地图BCE损失曲线
        if 'map_bce_loss' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'map_bce_loss', '地图BCE损失 (Map BCE Loss)', output_dir, title_suffix="变化曲线")
        
        # 绘制风险分数原始值曲线
        if 'risk_score_raw' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'risk_score_raw', '原始风险分数 (Raw Risk Score)', output_dir, title_suffix="变化曲线")
        
        # 绘制缩放后的风险分数曲线
        if 'risk_scaled' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'risk_scaled', '缩放风险分数 (Scaled Risk Score)', output_dir, title_suffix="变化曲线")
        
        # 绘制更多损失分量的曲线
        # 重构损失相关
        if 'risk_weight' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'risk_weight', '风险感知损失权重系数 (Risk Weight)', output_dir, title_suffix="变化曲线")
            
        # KL散度相关
        if 'risk_autoscale_target_frac' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'risk_autoscale_target_frac', 'KL散度目标分数 (KL Target Fraction)', output_dir, title_suffix="变化曲线")
            
        # 对抗接近性损失相关
        if 'compw_risk_min_dist' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'compw_risk_min_dist', '最小距离风险权重 (Min Dist Risk Weight)', output_dir, title_suffix="变化曲线")
            
        if 'compw_risk_ttc' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'compw_risk_ttc', 'TTC风险权重 (TTC Risk Weight)', output_dir, title_suffix="变化曲线")
            
        if 'compw_risk_pet' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'compw_risk_pet', 'PET风险权重 (PET Risk Weight)', output_dir, title_suffix="变化曲线")
            
        if 'compw_risk_overlap' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'compw_risk_overlap', '重叠风险权重 (Overlap Risk Weight)', output_dir, title_suffix="变化曲线")
            
        # 运动学约束损失相关
        if 'compw_entropy_lambda' in autoscale_df.columns:
            plot_metric_curves(autoscale_df, 'compw_entropy_lambda', '组件熵权重系数 (Component Entropy Lambda)', output_dir, title_suffix="变化曲线")
    
    print(f"\n所有曲线图已生成并保存到 '{output_dir}' 目录中。")

if __name__ == "__main__":
    main()
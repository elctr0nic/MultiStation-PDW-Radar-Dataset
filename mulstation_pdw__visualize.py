import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import os

# ================= 配置参数 =================
DATA_DIR = './datasets/generated_data/'
OUTPUT_DIR = './datasets/output_visualization/'
STATIONS = ['S1', 'S2', 'S3']
ZOOM_QUANTILES = (0.05, 0.95)

# 雷达颜色映射 (5部雷达 + 噪声)
RADAR_COLORS = {
    0.0: '#1f77b4',  # Blue - Radar 0
    1.0: '#ff7f0e',  # Orange - Radar 1
    2.0: '#2ca02c',  # Green - Radar 2
    3.0: '#d62728',  # Red - Radar 3
    4.0: '#9467bd',  # Purple - Radar 4
    -1.0: '#bfbfbf'  # Gray - Noise
}

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def load_data():
    """加载所有站点的CSV数据"""
    data = {}
    for s in STATIONS:
        path = os.path.join(DATA_DIR, f"{s}_data.csv")
        if os.path.exists(path):
            data[s] = pd.read_csv(path)
            print(f"✓ 加载 {s}: {len(data[s])} 脉冲")
        else:
            print(f"✗ 未找到: {path}")
    return data


def _scatter_by_label(ax, df_subset, y_col, with_labels=True, noise_size=15, signal_size=25):
    """Draw PDW points with the same Label-based color mapping."""
    noise = df_subset[df_subset['Label'] == -1]
    noise_label = 'Noise' if with_labels else '_nolegend_'
    if len(noise) > 0:
        ax.scatter(noise['TOA'], noise[y_col],
                   c=RADAR_COLORS[-1], s=noise_size, alpha=0.4,
                   label=noise_label, edgecolors='none')

    signal = df_subset[df_subset['Label'] != -1]
    for radar_id in sorted(signal['Label'].unique()):
        subset = signal[signal['Label'] == radar_id]
        radar_label = f'Radar {int(radar_id)}' if with_labels else '_nolegend_'
        ax.scatter(subset['TOA'], subset[y_col],
                   c=RADAR_COLORS.get(radar_id, 'black'), s=signal_size, alpha=0.7,
                   label=radar_label, edgecolors='none')


def _padded_limits(low, high, pad_ratio=0.08):
    if not np.isfinite(low) or not np.isfinite(high):
        return None

    span = high - low
    if span <= 0:
        margin = max(abs(high) * pad_ratio, 1.0)
    else:
        margin = span * pad_ratio
    return low - margin, high + margin


def _add_dense_zoom_inset(ax, df_subset, y_col, title):
    """Add an inset focused on the central signal distribution."""
    zoom_source = df_subset[df_subset['Label'] != -1]
    if len(zoom_source) < 3:
        zoom_source = df_subset
    if len(zoom_source) < 3:
        return

    q_low, q_high = ZOOM_QUANTILES
    toa_low, toa_high = zoom_source['TOA'].quantile([q_low, q_high])
    y_low, y_high = zoom_source[y_col].quantile([q_low, q_high])

    x_limits = _padded_limits(toa_low, toa_high)
    y_limits = _padded_limits(y_low, y_high)
    if x_limits is None or y_limits is None:
        return

    zoom_ax = inset_axes(ax, width="40%", height="48%", loc='lower right', borderpad=1.2)
    _scatter_by_label(zoom_ax, df_subset, y_col, with_labels=False, noise_size=8, signal_size=12)
    zoom_ax.set_xlim(*x_limits)
    zoom_ax.set_ylim(*y_limits)
    zoom_ax.set_title(title, fontsize=8, fontweight='bold')
    zoom_ax.grid(True, linestyle='--', alpha=0.35)
    zoom_ax.tick_params(axis='both', labelsize=7)
    mark_inset(ax, zoom_ax, loc1=2, loc2=4, fc="none", ec="0.35", lw=0.8)


def plot_toa_rf_per_station(data_map):
    """
    生成每个站点的 TOA-RF 二维图
    时间轴 vs 频率轴，按雷达ID着色
    """
    print("\n>>> 生成 TOA-RF 图...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time_limit = 200000  # 前 200ms
    
    for idx, station in enumerate(STATIONS):
        df = data_map[station]
        df_subset = df[df['TOA'] < time_limit].copy()
        
        ax = axes[idx]
        _scatter_by_label(ax, df_subset, 'RF')
        
        ax.set_ylabel(f"RF (MHz)", fontsize=11, fontweight='bold')
        ax.set_title(f"Station {station} - TOA vs RF Distribution", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        _add_dense_zoom_inset(ax, df_subset, 'RF', 'Dense RF zoom')
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=6, fontsize=9, framealpha=0.9)
    
    axes[-1].set_xlabel("Time of Arrival (us)", fontsize=11, fontweight='bold')
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.35)
    output_path = os.path.join(OUTPUT_DIR, '1_TOA_RF_per_station.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_path}")
    plt.close()


def plot_toa_pw_per_station(data_map):
    """
    生成每个站点的 TOA-PW 二维图
    时间轴 vs 脉冲宽度轴
    """
    print(">>> 生成 TOA-PW 图...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time_limit = 200000
    
    for idx, station in enumerate(STATIONS):
        df = data_map[station]
        df_subset = df[df['TOA'] < time_limit].copy()
        
        ax = axes[idx]
        _scatter_by_label(ax, df_subset, 'PW')
        
        ax.set_ylabel(f"PW (μs)", fontsize=11, fontweight='bold')
        ax.set_title(f"Station {station} - TOA vs PW Distribution", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        _add_dense_zoom_inset(ax, df_subset, 'PW', 'Dense PW zoom')
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=6, fontsize=9, framealpha=0.9)
    
    axes[-1].set_xlabel("Time of Arrival (us)", fontsize=11, fontweight='bold')
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.07, hspace=0.35)
    output_path = os.path.join(OUTPUT_DIR, '2_TOA_PW_per_station.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_path}")
    plt.close()


def plot_toa_pa_per_station(data_map):
    """
    生成每个站点的 TOA-PA 二维图
    时间轴 vs 接收功率轴 (dBm)
    """
    print(">>> 生成 TOA-PA 图...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time_limit = 200000
    
    for idx, station in enumerate(STATIONS):
        df = data_map[station]
        df_subset = df[df['TOA'] < time_limit].copy()
        
        ax = axes[idx]
        
        # 噪声
        noise = df_subset[df_subset['Label'] == -1]
        if len(noise) > 0:
            ax.scatter(noise['TOA'], noise['PA'],
                      c=RADAR_COLORS[-1], s=15, alpha=0.4,
                      label='Noise', edgecolors='none')
        
        # 真实信号
        for radar_id in sorted(df_subset[df_subset['Label'] != -1]['Label'].unique()):
            subset = df_subset[df_subset['Label'] == radar_id]
            ax.scatter(subset['TOA'], subset['PA'],
                      c=RADAR_COLORS.get(radar_id, 'black'), s=25, alpha=0.7,
                      label=f'Radar {int(radar_id)}', edgecolors='none')
        
        ax.set_ylabel(f"PA (dBm)", fontsize=11, fontweight='bold')
        ax.set_title(f"Station {station} - TOA vs PA Distribution", fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=6, fontsize=9, framealpha=0.9)
    
    axes[-1].set_xlabel("Time of Arrival (us)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '3_TOA_PA_per_station.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_path}")
    plt.close()


def plot_toa_doa_per_station(data_map):
    """
    生成每个站点的 TOA-DOA 二维图
    时间轴 vs 方位角轴 (度)
    """
    print(">>> 生成 TOA-DOA 图...")
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time_limit = 200000
    
    for idx, station in enumerate(STATIONS):
        df = data_map[station]
        df_subset = df[df['TOA'] < time_limit].copy()
        
        ax = axes[idx]
        
        # 噪声
        noise = df_subset[df_subset['Label'] == -1]
        if len(noise) > 0:
            ax.scatter(noise['TOA'], noise['DOA'],
                      c=RADAR_COLORS[-1], s=15, alpha=0.4,
                      label='Noise', edgecolors='none')
        
        # 真实信号
        for radar_id in sorted(df_subset[df_subset['Label'] != -1]['Label'].unique()):
            subset = df_subset[df_subset['Label'] == radar_id]
            ax.scatter(subset['TOA'], subset['DOA'],
                      c=RADAR_COLORS.get(radar_id, 'black'), s=25, alpha=0.7,
                      label=f'Radar {int(radar_id)}', edgecolors='none')
        
        ax.set_ylabel(f"DOA (°)", fontsize=11, fontweight='bold')
        ax.set_title(f"Station {station} - TOA vs DOA Distribution", fontsize=12, fontweight='bold')
        ax.set_ylim(-180, 180)  # 方位角范围
        ax.grid(True, linestyle='--', alpha=0.5)
        
        if idx == 0:
            ax.legend(loc='upper right', ncol=6, fontsize=9, framealpha=0.9)
    
    axes[-1].set_xlabel("Time of Arrival (us)", fontsize=11, fontweight='bold')
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '4_TOA_DOA_per_station.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_path}")
    plt.close()


def plot_3d_rf_pw_pa_per_station(data_map):
    """
    生成每个站点的 RF-PW-PA 三维散点图
    X轴: RF (MHz)
    Y轴: PW (μs)
    Z轴: PA (dBm)
    """
    print(">>> 生成 RF-PW-PA 三维图...")
    
    fig = plt.figure(figsize=(16, 13))
    
    for idx, station in enumerate(STATIONS, 1):
        df = data_map[station]
        
        # 采样以避免过拥挤 (特别是噪声)
        df_signal = df[df['Label'] != -1]
        df_noise = df[df['Label'] == -1]
        
        # 如果噪声太多，随机采样
        if len(df_noise) > 2000:
            df_noise = df_noise.sample(2000, random_state=42)
        
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # 绘制噪声
        if len(df_noise) > 0:
            ax.scatter(df_noise['RF'], df_noise['PW'], df_noise['PA'],
                      c=RADAR_COLORS[-1], s=10, alpha=0.2,
                      label='Noise', edgecolors='none')
        
        # 绘制真实信号
        for radar_id in sorted(df_signal['Label'].unique()):
            subset = df_signal[df_signal['Label'] == radar_id]
            ax.scatter(subset['RF'], subset['PW'], subset['PA'],
                      c=RADAR_COLORS.get(radar_id, 'black'), s=40, alpha=0.8,
                      label=f'Radar {int(radar_id)}', edgecolors='none')
        
        ax.set_xlabel('RF (MHz)', fontsize=10, fontweight='bold')
        ax.set_ylabel('PW (μs)', fontsize=10, fontweight='bold')
        ax.set_zlabel('PA (dBm)', fontsize=10, fontweight='bold')
        ax.set_title(f"Station {station} - 3D Feature Space", fontsize=11, fontweight='bold')
        
        # 图例
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    # 空出第4个位置用于整体图例
    ax_legend = fig.add_subplot(2, 2, 4)
    ax_legend.axis('off')
    
    # 添加统计信息
    info_text = "Statistics:\n"
    for station in STATIONS:
        df = data_map[station]
        n_total = len(df)
        n_signal = len(df[df['Label'] != -1])
        n_noise = len(df[df['Label'] == -1])
        info_text += f"\n{station}: {n_total} pulses\n  Signal: {n_signal}\n  Noise: {n_noise}"
    
    ax_legend.text(0.1, 0.5, info_text, fontsize=10, family='monospace',
                   verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '5_3D_RF_PW_PA_per_station.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_path}")
    plt.close()


def plot_feature_statistics(data_map):
    """
    生成每个站点的特征统计图
    显示 RF, PW, PA, DOA 的分布直方图
    """
    print(">>> 生成特征统计图...")
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    features = ['RF', 'PW', 'PA', 'DOA']
    bins = {'RF': 30, 'PW': 30, 'PA': 30, 'DOA': 36}
    
    for row, station in enumerate(STATIONS):
        df_signal = data_map[station][data_map[station]['Label'] != -1]
        
        for col, feature in enumerate(features):
            ax = axes[row, col]
            
            # 绘制每个雷达的直方图
            for radar_id in sorted(df_signal['Label'].unique()):
                subset = df_signal[df_signal['Label'] == radar_id]
                ax.hist(subset[feature], bins=bins[feature], alpha=0.6,
                       label=f'Radar {int(radar_id)}',
                       color=RADAR_COLORS.get(radar_id, 'black'))
            
            ax.set_xlabel(feature, fontsize=10, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax.set_title(f"{station} - {feature} Distribution", fontsize=11, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.5)
            
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, '6_Feature_Statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存: {output_path}")
    plt.close()


def generate_summary_report(data_map):
    """生成数据汇总报告"""
    print("\n" + "="*60)
    print("数据汇总统计报告")
    print("="*60)
    
    for station in STATIONS:
        df = data_map[station]
        print(f"\n★ Station {station}:")
        print(f"  总脉冲数: {len(df)}")
        print(f"  真实信号: {len(df[df['Label'] != -1])}")
        print(f"  杂散脉冲: {len(df[df['Label'] == -1])}")
        print(f"  特征范围:")
        print(f"    RF:  [{df['RF'].min():.1f}, {df['RF'].max():.1f}] MHz")
        print(f"    PW:  [{df['PW'].min():.2f}, {df['PW'].max():.2f}] μs")
        print(f"    PA:  [{df['PA'].min():.1f}, {df['PA'].max():.1f}] dBm")
        print(f"    DOA: [{df['DOA'].min():.1f}, {df['DOA'].max():.1f}] °")
        
        # 按雷达统计
        for radar_id in sorted(df[df['Label'] != -1]['Label'].unique()):
            count = len(df[df['Label'] == radar_id])
            print(f"    Radar {int(radar_id)}: {count} pulses")


def main():
    print("\n" + "="*60)
    print("PDW多站点可视化工具")
    print("="*60)
    
    # 加载数据
    data = load_data()
    if not data:
        print("错误: 没有可用的数据文件!")
        return
    
    print(f"\n✓ 数据加载完成，共加载 {len(data)} 个站点")
    
    # 生成所有可视化图表
    print("\n>>> 开始生成可视化图表...\n")
    
    plot_toa_rf_per_station(data)
    plot_toa_pw_per_station(data)
    plot_toa_pa_per_station(data)
    plot_toa_doa_per_station(data)
    plot_3d_rf_pw_pa_per_station(data)
    plot_feature_statistics(data)
    
    # 生成统计报告
    generate_summary_report(data)
    
    print("\n" + "="*60)
    print(f"✓ 所有图表已生成！")
    print(f"  保存位置: {os.path.abspath(OUTPUT_DIR)}")
    print("="*60 + "\n")
    
    # 列出生成的文件
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    print("生成的文件列表:")
    for i, f in enumerate(sorted(files), 1):
        print(f"  {i}. {f}")


if __name__ == "__main__":
    main()

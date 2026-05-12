from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


C_LIGHT_KM_S = 3e5
C_LIGHT_M_S = 3e8
DEFAULT_TOTAL_TIME_US = 2_000 * 1e3
DEFAULT_OUTPUT_DIR = Path("./datasets/generated_data")

STATION_COORDS: Dict[str, np.ndarray] = {
    "S1": np.array([20.0, 0.0, 0.0], dtype=float),
    "S2": np.array([0.0, -100.0, 0.0], dtype=float),
    "S3": np.array([0.0, 10.0, -10.0], dtype=float),
}

BASE_STATION_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "S1": {
        "coord": STATION_COORDS["S1"].copy(),          # 基站坐标（拷贝防止篡改原数据）
        "receiver_mode": "stare",                      # 接收模式：凝视（持续监听）
        "noise_floor_dbm": -96.0,                      # 噪声底（dBm，信号噪声强度）
        "threshold_snr_db": 4.0,                      # 信噪比阈值（低于则不识别信号）
        "scan_cycle_us": 0.0,                          # 扫描周期（凝视模式无扫描）
        "scan_dwell_us": 0.0,                          # 扫描驻留时间
        "scan_phase_us": 0.0,                          # 扫描相位
        # 信号测量分辨率
        "resolution": {
            "toa_us": 0.05,        # 到达时间分辨率（微秒）
            "rf_mhz": 1.0,         # 射频频率分辨率（兆赫）
            "pw_us": 0.1,          # 脉冲宽度分辨率（微秒）
            "pa_db": 0.5,          # 功率幅度分辨率（分贝）
        },
        # 测量噪声底（测量误差）
        "measurement_noise_floor": {
            "rf_mhz": 0.2,
            "pw_us": 0.04,
            "pa_db": 0.3,
            "doa_deg": 0.8,        # 波达方向误差（度）
        },
    },
    # S2基站：扫描模式（周期性扫描，非持续监听）
    "S2": {
        "coord": STATION_COORDS["S2"].copy(),
        "receiver_mode": "scan",
        "noise_floor_dbm": -95.0,
        "threshold_snr_db": 4.5,
        "scan_cycle_us": 1_200.0,   # 扫描周期1200微秒
        "scan_dwell_us": 720.0,     # 扫描驻留720微秒
        "scan_phase_us": 140.0,     # 扫描相位140微秒
        "resolution": {
            "toa_us": 0.08,
            "rf_mhz": 1.5,
            "pw_us": 0.15,
            "pa_db": 0.5,
        },
        "measurement_noise_floor": {
            "rf_mhz": 0.3,
            "pw_us": 0.05,
            "pa_db": 0.35,
            "doa_deg": 1.0,
        },
    },
    # S3基站：凝视模式，参数与S1略有差异
    "S3": {
        "coord": STATION_COORDS["S3"].copy(),
        "receiver_mode": "stare",
        "noise_floor_dbm": -97.0,
        "threshold_snr_db": 3.5,
        "scan_cycle_us": 0.0,
        "scan_dwell_us": 0.0,
        "scan_phase_us": 0.0,
        "resolution": {
            "toa_us": 0.04,
            "rf_mhz": 0.8,
            "pw_us": 0.08,
            "pa_db": 0.4,
        },
        "measurement_noise_floor": {
            "rf_mhz": 0.15,
            "pw_us": 0.03,
            "pa_db": 0.25,
            "doa_deg": 0.7,
        },
    },
}
BASE_EMITTER_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": 0,                                # 发射源唯一ID
        "position": np.array([100.0, 120.0, 50.0], dtype=float), # 三维位置
        "label": 0,                              # 分类标签
        "pri_type": "jitter",                    # 脉冲重复间隔类型：抖动
        "pri_base": 70.0,                        # 基础PRI（脉冲间隔70微秒）
        "pri_jitter_pct": 0.20,                  # PRI抖动比例20%
        "rf_type": "group",                      # 射频类型：分组
        "rf_range": [8_000.0, 9_400.0],          # 射频频率范围（MHz）
        "pw_type": "slide",                      # 脉冲宽度类型：滑动
        "pw_range": [15.0, 30.0],                # 脉冲宽度范围（微秒）
        "pw_jitter_pct": 0.08,                   # 脉宽抖动比例8%
        "tx_power_dbm": 50.0,                    # 发射功率（dBm）
        "shadow_sigma_db": 1.2,                  # 信号阴影衰落标准差
    },
    # 发射源1：PRI交错、射频跳变、脉宽抖动
    {
        "id": 1,
        "position": np.array([80.0, 210.0, 150.0], dtype=float),
        "label": 1,
        "pri_type": "stagger",
        "pri_list": [125.0, 162.0, 100.0],
        "pri_jitter_pct": 0.03,
        "rf_type": "agile",
        "rf_range": [7_800.0, 8_500.0],
        "pw_type": "jitter",
        "pw_base": 22.0,
        "pw_jitter_pct": 0.12,
        "tx_power_dbm": 45.0,
        "shadow_sigma_db": 1.5,
    },
    # 发射源2：PRI滑动、脉宽固定
    {
        "id": 2,
        "position": np.array([130.0, 146.0, 280.0], dtype=float),
        "label": 2,
        "pri_type": "slide",
        "pri_range": [85.0, 160.0],
        "pri_jitter_pct": 0.03,
        "rf_type": "group",
        "rf_range": [7_600.0, 8_700.0],
        "pw_type": "fixed",
        "pw_base": 20.0,
        "pw_jitter_pct": 0.04,
        "tx_power_dbm": 48.0,
        "shadow_sigma_db": 1.0,
    },
    # 发射源3：PRI固定、射频跳变
    {
        "id": 3,
        "position": np.array([150.0, 30.0, 115.0], dtype=float),
        "label": 3,
        "pri_type": "fixed",
        "pri_base": 40.0,
        "pri_jitter_pct": 0.03,
        "rf_type": "agile",
        "rf_range": [8_200.0, 9_200.0],
        "pw_type": "jitter",
        "pw_base": 18.00,
        "pw_jitter_pct": 0.12,
        "tx_power_dbm": 52.0,
        "shadow_sigma_db": 1.3,
    },
    # 发射源4：PRI交错、脉宽滑动
    {
        "id": 4,
        "position": np.array([100.0, 102.0, 150.0], dtype=float),
        "label": 4,
        "pri_type": "stagger",
        "pri_list": [80.0, 100.0, 130.0],
        "pri_jitter_pct": 0.03,
        "rf_type": "group",
        "rf_range": [8_400.0, 9_500.0],
        "pw_type": "slide",
        "pw_range": [23.0, 35.0],
        "pw_jitter_pct": 0.06,
        "tx_power_dbm": 46.0,
        "shadow_sigma_db": 1.1,
    },
]
DIFFICULTY_PRESETS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "snr_bias_db": [10.0, 14.0],            # 信噪比偏高（信号清晰）
        "time_sync_toa_error_ns": [20.0, 30.0], # 时间同步误差小
        "receiver_miss_probability": [0.01, 0.03],# 信号丢失概率低
        "burst_loss_strength": [0.01, 0.03],     # 脉冲串丢失强度低
        "structured_interference_ratio": [0.05, 0.10],# 干扰比例低
        "emitter_count": [5, 5],                 # 发射源数量4-5个
    },
    "medium": {
        "snr_bias_db": [4.0, 7.0],               # 信噪比中等
        "time_sync_toa_error_ns": [30.0, 40.0],
        "receiver_miss_probability": [0.04, 0.08],
        "burst_loss_strength": [0.04, 0.08],
        "structured_interference_ratio": [0.12, 0.18],
        "emitter_count": [5, 5],                 # 固定5个发射源
    },
    "hard": {
        "snr_bias_db": [-2.0, 2.0],              # 信噪比极低（信号模糊）
        "time_sync_toa_error_ns": [40.0, 50.0],
        "receiver_miss_probability": [0.10, 0.16],# 信号易丢失
        "burst_loss_strength": [0.10, 0.18],
        "structured_interference_ratio": [0.20, 0.32],# 强干扰
        "emitter_count": [5, 5],
    },
    "difficult": {
        "snr_bias_db": [-8.0,-52.0],              # 信噪比极低（信号模糊）
        "time_sync_toa_error_ns": [60.0, 70.0],
        "receiver_miss_probability": [0.20, 0.28],# 信号易丢失
        "burst_loss_strength": [0.20, 0.30],
        "structured_interference_ratio": [0.30, 0.38],# 强干扰
        "emitter_count": [6, 6],
    },
}


def get_station_names() -> List[str]:
    return list(STATION_COORDS.keys())


def get_station_coords() -> Dict[str, np.ndarray]:
    return {name: coord.copy() for name, coord in STATION_COORDS.items()}


def get_default_output_dir() -> Path:
    return DEFAULT_OUTPUT_DIR


def _sample_uniform(rng: np.random.Generator, value_range: List[float]) -> float:
    # 输入：随机数生成器 + 数值范围 [最小值, 最大值]
    # 输出：范围内的随机浮点数
    return float(rng.uniform(value_range[0], value_range[1]))


def _sample_int(rng: np.random.Generator, value_range: List[int]) -> int:
    low, high = int(value_range[0]), int(value_range[1])
    # 输入：随机数生成器 + 整数范围 [最小值, 最大值]
    # 输出：范围内的随机整数（包含上下限）
    return int(rng.integers(low, high + 1))


def _build_station_configs(
    rng: np.random.Generator,  # 随机数生成器
    axes: Dict[str, Any],      # 难度参数
    total_time_us: float,      # 仿真时长
) -> Dict[str, Dict[str, Any]]:
    stations: Dict[str, Dict[str, Any]] = {}  # 存储最终基站配置
    # 遍历每个基站模板
    for station_id, template in BASE_STATION_TEMPLATES.items():
        station_cfg = deepcopy(template)  # 深拷贝模板，不修改原始数据
        station_cfg["coord"] = np.array(station_cfg["coord"], dtype=float)  # 坐标转浮点数组
        station_cfg["station_id"] = station_id  # 绑定基站ID
        # 采样时间同步误差（纳秒）
        station_cfg["time_sync_toa_error_ns"] = _sample_uniform(rng, axes["time_sync_toa_error_ns"])
        # 转换为微秒（1微秒=1000纳秒）
        station_cfg["time_sync_toa_error_us"] = station_cfg["time_sync_toa_error_ns"] * 1e-3
        station_cfg["time_sync_tdoa_budget_us"] = station_cfg["time_sync_toa_error_us"]
        # 采样信号丢失概率
        station_cfg["receiver_miss_probability"] = _sample_uniform(
            rng, axes["receiver_miss_probability"]
        )
        # 采样脉冲串丢失强度
        station_cfg["burst_loss_strength"] = _sample_uniform(rng, axes["burst_loss_strength"])
        # 如果是扫描模式：添加随机相位抖动
        if station_cfg["receiver_mode"] == "scan":
            phase_jitter = 0.2 * station_cfg["scan_cycle_us"]
            station_cfg["scan_phase_us"] += float(rng.uniform(-phase_jitter, phase_jitter))
        stations[station_id] = station_cfg  # 存入最终配置
    return stations


def _pick_active_emitters(
    rng: np.random.Generator,
    axes: Dict[str, Any],
) -> List[Dict[str, Any]]:
    # 采样本次仿真的发射源数量
    emitter_count = _sample_int(rng, axes["emitter_count"])
    # 随机不重复选择发射源索引
    indices = sorted(rng.choice(len(BASE_EMITTER_TEMPLATES), size=emitter_count, replace=False))
    # 深拷贝选中的发射源模板
    emitters = [deepcopy(BASE_EMITTER_TEMPLATES[idx]) for idx in indices]
    # 为每个发射源补充参数
    for emitter in emitters:
        emitter["position"] = np.array(emitter["position"], dtype=float)
        emitter["visible_stations"] = get_station_names()  # 可见的基站
        emitter["emitter_dropout_probability"] = float(rng.uniform(0.0, 0.02))  # 发射源掉线概率
    return emitters


def get_default_simulation_config(
    difficulty: str = "medium",  # 默认难度：中等
    *,
    seed: int = 42,              # 随机种子（保证结果可复现）
    output_dir: str | Path | None = None,  # 输出目录
    total_time_us: float = DEFAULT_TOTAL_TIME_US,  # 仿真时长
) -> Dict[str, Any]:
    # 校验难度是否合法
    if difficulty not in DIFFICULTY_PRESETS:
        raise ValueError(f"Unsupported difficulty '{difficulty}'.")

    # 初始化随机数生成器（固定种子，结果可复现）
    rng = np.random.default_rng(seed)
    # 深拷贝难度预设参数
    axes_template = deepcopy(DIFFICULTY_PRESETS[difficulty])
    # 构建本次仿真的核心参数轴
    axes: Dict[str, Any] = {
        "average_snr_db": _sample_uniform(rng, axes_template["snr_bias_db"]),
        "station_time_sync_error_ns": None,
        "receiver_miss_probability": _sample_uniform(
            rng, axes_template["receiver_miss_probability"]
        ),
        "burst_loss_strength": _sample_uniform(rng, axes_template["burst_loss_strength"]),
        "structured_interference_ratio": _sample_uniform(
            rng, axes_template["structured_interference_ratio"]
        ),
        "emitter_count": _sample_int(rng, axes_template["emitter_count"]),
    }

    # 构建基站配置
    stations = _build_station_configs(rng, axes_template, total_time_us)
    # 选择活跃发射源
    emitters = _pick_active_emitters(rng, axes_template)

    # 校准发射源数量
    if axes["emitter_count"] < len(emitters):
        emitters = emitters[: axes["emitter_count"]]
    elif axes["emitter_count"] > len(emitters):
        axes["emitter_count"] = len(emitters)


    # 计算最大时间同步误差
    max_time_sync_error_ns = max(station["time_sync_toa_error_ns"] for station in stations.values())
    axes["station_time_sync_error_ns"] = max_time_sync_error_ns

    # 生成场景ID
    scenario_id = f"pdw_{difficulty}_seed{seed}"
    # 确定输出目录
    resolved_output_dir = Path(output_dir) if output_dir is not None else DEFAULT_OUTPUT_DIR

    # 返回**完整的仿真配置字典**（最终输出）
    return {
        "scenario_id": scenario_id,
        "difficulty": difficulty,
        "seed": seed,
        "total_time_us": float(total_time_us),
        "output_dir": str(resolved_output_dir),
        "axes": axes,
        "global_channel": {  # 全局信道参数
            "tx_gain_dbi": 20.0,
            "rx_gain_dbi": 20.0,
            "cable_loss_db": 2.0,
            "snr_bias_db": axes["average_snr_db"],
        },
        "stations": stations,  # 基站配置
        "emitters": emitters,  # 发射源配置
        "interference": {      # 干扰配置
            "structured_interference_ratio": axes["structured_interference_ratio"],
            "thermal_share": 0.30,# 热噪声占比
            "cochannel_share": 0.25,# 同频干扰占比
            "cluster_share": 0.20,# 群集干扰占比
            "decoy_share": 0.25,# 诱饵干扰占比
        },
    }

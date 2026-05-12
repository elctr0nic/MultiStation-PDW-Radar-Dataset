# 启用未来版本的类型注解语法，兼容低版本Python
from __future__ import annotations

#  JSON序列化、日志、路径处理、类型注解
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 数值计算、表格数据处理
import numpy as np
import pandas as pd

# 尝试导入配置文件（两种路径兼容，防止导入失败）
try:
    from pdw_sim_config import (
        C_LIGHT_KM_S,        # 光速（公里/秒）
        C_LIGHT_M_S,        # 光速（米/秒）
        DEFAULT_TOTAL_TIME_US, # 默认仿真时长
        get_default_simulation_config, # 获取仿真配置
    )
except ImportError:
    from datasets.pdw_sim_config import (
        C_LIGHT_KM_S,
        C_LIGHT_M_S,
        DEFAULT_TOTAL_TIME_US,
        get_default_simulation_config,
    )


# 配置日志输出格式：时间-级别-信息，级别为INFO
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)

# 最终输出的CSV文件列名（PDW核心特征）
OBSERVATION_COLUMNS = [
    "TOA",          # 脉冲到达时间
    "RF",           # 射频频率
    "PW",           # 脉冲宽度
    "PA",           # 功率幅度
    "DOA",          # 波达方向
    "Label",        # 发射源标签（分类用）
    "SNR",          # 信噪比
    "station_id",   # 基站ID
    "scenario_id",  # 仿真场景ID
    "receiver_mode",# 接收机模式（凝视/扫描）
    "emitter_id",   # 发射源ID
    "resolution_bin_toa",  # TOA量化索引
    "resolution_bin_rf",   # RF量化索引
    "resolution_bin_pw",   # PW量化索引
    "resolution_bin_pa",   # PA量化索引
]

def quantize_values(values: np.ndarray, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    # 防止分辨率为0，避免除零错误
    resolution = max(float(resolution), 1e-9)
    # 四舍五入到分辨率整数倍，生成量化索引
    bins = np.rint(values / resolution).astype(np.int64)
    # 返回量化后的值 + 量化索引
    return bins.astype(np.float64) * resolution, bins

def to_serializable(value: Any) -> Any:
    # 把numpy数组、路径、数值等转为JSON可保存的普通类型
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    return value


# 根据发射源配置，生成原始雷达脉冲真值（无噪声、无传播损耗
class EmitterModel:
    # 初始化：传入发射源配置、仿真时长、随机种子
    def __init__(self, emitters: List[Dict[str, Any]], total_time_us: float, seed: int) -> None:
        self.emitters = emitters          # 所有发射源
        self.total_time_us = float(total_time_us) # 仿真总时长
        self.seed = int(seed)             # 随机种子（保证可复现）

    # 为每个发射源生成独立随机数生成器
    def _emitter_rng(self, emitter_id: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + emitter_id * 10_003)

    # 采样脉冲重复间隔（PRI）：决定脉冲发射的时间间隔
    def _sample_pri(self, emitter: Dict[str, Any], pulse_index: int, rng: np.random.Generator) -> float:
        pri_type = emitter["pri_type"]
        # 固定PRI
        if pri_type == "fixed":
            base_pri = emitter["pri_base"]
        # 抖动PRI
        elif pri_type == "jitter":
            base_pri = emitter["pri_base"]
        # 交错PRI（循环切换多个值）
        elif pri_type == "stagger":
            pri_list = emitter["pri_list"]
            base_pri = pri_list[pulse_index % len(pri_list)]
        # 滑动PRI（线性递增/递减）
        elif pri_type == "slide":
            start, end = emitter["pri_range"]
            cycle_len = 50
            step = (end - start) / max(cycle_len - 1, 1)
            base_pri = start + (pulse_index % cycle_len) * step
        else:
            base_pri = emitter.get("pri_base", 100.0)
        # 添加PRI抖动
        jitter_pct = float(emitter.get("pri_jitter_pct", 0.0))
        return max(0.1, base_pri * (1.0 + rng.uniform(-jitter_pct, jitter_pct)))

    # 采样射频频率（RF）
    def _sample_rf(self, emitter: Dict[str, Any], pulse_index: int, rng: np.random.Generator) -> float:
        rf_type = emitter["rf_type"]
        # 跳频（每脉冲随机）
        if rf_type == "agile":
            return float(rng.uniform(*emitter["rf_range"]))
        # 分组跳频（8个脉冲一组，频率相同）
        if rf_type == "group":
            group_idx = pulse_index // 8
            group_rng = np.random.default_rng(self.seed + emitter["id"] * 97 + group_idx)
            return float(group_rng.uniform(*emitter["rf_range"]))
        # 固定频率
        if rf_type == "fixed":
            return float(emitter["rf_base"])
        return float(emitter.get("rf_base", np.mean(emitter.get("rf_range", [8_000.0, 8_200.0]))))

    # 采样脉冲宽度（PW）
    def _sample_pw(self, emitter: Dict[str, Any], pulse_index: int, rng: np.random.Generator) -> float:
        pw_type = emitter["pw_type"]
        # 固定脉宽
        if pw_type == "fixed":
            base_pw = float(emitter["pw_base"])
        # 抖动脉宽
        elif pw_type == "jitter":
            base_pw = float(emitter["pw_base"])
        # 滑动脉宽
        elif pw_type == "slide":
            start, end = emitter["pw_range"]
            cycle_len = 20
            step = (end - start) / max(cycle_len - 1, 1)
            base_pw = start + (pulse_index % cycle_len) * step
        else:
            base_pw = float(emitter.get("pw_base", np.mean(emitter.get("pw_range", [10.0, 10.0]))))
        # 添加PW抖动
        jitter_pct = float(emitter.get("pw_jitter_pct", 0.0))
        return max(0.05, base_pw * (1.0 + rng.uniform(-jitter_pct, jitter_pct)))

    # 生成所有发射源的脉冲序列（主函数）
    def generate(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        all_frames: List[pd.DataFrame] = []
        emitter_stats: List[Dict[str, Any]] = []

        # 遍历每个发射源，生成脉冲
        for emitter in self.emitters:
            emitter_id = int(emitter["id"])
            rng = self._emitter_rng(emitter_id)
            rows: List[Dict[str, Any]] = []
            pulse_index = 0
            emitted_count = 0
            dropped_count = 0
            time_us = 0.0

            # 循环生成脉冲，直到超过仿真时长
            while time_us < self.total_time_us:
                # 采样PRI，更新发射时间
                time_us += self._sample_pri(emitter, pulse_index, rng)
                if time_us >= self.total_time_us:
                    break
                # 采样RF、PW
                rf_mhz = self._sample_rf(emitter, pulse_index, rng)
                pw_us = self._sample_pw(emitter, pulse_index, rng)
                pulse_id = f"E{emitter_id:02d}_P{pulse_index:07d}"

                # 模拟发射源掉线（随机丢弃脉冲）
                if rng.random() < float(emitter.get("emitter_dropout_probability", 0.0)):
                    dropped_count += 1
                    pulse_index += 1
                    continue

                # 保存脉冲真值数据
                rows.append(
                    {
                        "pulse_id": pulse_id,
                        "pulse_index": pulse_index,
                        "emitter_id": emitter_id,
                        "Label": int(emitter["label"]),
                        "toa_emission_us": time_us,
                        "rf_true_mhz": rf_mhz,
                        "pw_true_us": pw_us,
                        "tx_power_dbm": float(emitter["tx_power_dbm"]),
                    }
                )
                emitted_count += 1
                pulse_index += 1

            # 转为DataFrame，统计发射源数据
            frame = pd.DataFrame(rows)
            if not frame.empty:
                all_frames.append(frame)

            emitter_stats.append(
                {
                    "emitter_id": emitter_id,
                    "label": int(emitter["label"]),
                    "attempted_pulses": emitted_count + dropped_count,
                    "emitted_pulses": emitted_count,
                    "emitter_dropout_count": dropped_count,
                    "receiver_visibility": emitter.get("visible_stations", []),
                }
            )

        # 合并所有脉冲，按发射时间排序
        if not all_frames:
            return pd.DataFrame(), emitter_stats
        pulses = pd.concat(all_frames, ignore_index=True).sort_values("toa_emission_us").reset_index(drop=True)
        return pulses, emitter_stats

#模拟雷达信号空间传播，计算时延、路径损耗、接收功率、SNR、波达方向 (DOA)
class ChannelModel:
    def __init__(self, scenario_cfg: Dict[str, Any], seed: int) -> None:
        self.scenario_cfg = scenario_cfg
        self.channel_cfg = scenario_cfg["global_channel"] # 全局信道配置
        self.seed = int(seed)

    # 为每个发射源+基站生成独立随机数
    def _channel_rng(self, emitter_id: int, station_name: str) -> np.random.Generator:
        station_hash = sum(ord(char) for char in station_name)
        return np.random.default_rng(self.seed + emitter_id * 31 + station_hash * 101)

    # 信号传播计算（主函数）
    def propagate(
        self,
        pulses: pd.DataFrame,
        emitter_cfg: Dict[str, Any],
        station_name: str,
        station_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        if pulses.empty:
            return pd.DataFrame()
        # 发射源对该基站不可见，直接返回空
        if station_name not in emitter_cfg.get("visible_stations", []):
            return pd.DataFrame()

        rng = self._channel_rng(int(emitter_cfg["id"]), station_name)
        # 计算发射源与基站的距离、传播时延
        emitter_pos = np.asarray(emitter_cfg["position"], dtype=float)
        station_pos = np.asarray(station_cfg["coord"], dtype=float)
        rel_vec = emitter_pos - station_pos
        dist_km = float(np.linalg.norm(rel_vec)) # 欧式距离（公里）
        delay_us = (dist_km / C_LIGHT_KM_S) * 1e6 # 传播时延（微秒）

        # 计算自由空间路径损耗（雷达核心公式）
        rf_values = pulses["rf_true_mhz"].to_numpy(dtype=float)
        d_m = dist_km * 1_000.0
        freq_hz = rf_values * 1e6
        loss_linear = np.maximum((4.0 * np.pi * d_m * freq_hz / C_LIGHT_M_S) ** 2, 1e-12)
        path_loss_db = 10.0 * np.log10(loss_linear)
        # 阴影衰落（模拟环境干扰）
        shadow_fading = rng.normal(0.0, float(emitter_cfg.get("shadow_sigma_db", 1.0)), size=len(pulses))

        # 计算接收功率、信噪比
        rx_power_dbm = (
            pulses["tx_power_dbm"].to_numpy(dtype=float)
            + float(self.channel_cfg["tx_gain_dbi"])    # 发射增益
            + float(self.channel_cfg["rx_gain_dbi"])    # 接收增益
            - path_loss_db                               # 路径损耗
            - float(self.channel_cfg["cable_loss_db"])   # 线缆损耗
            + shadow_fading                             # 阴影衰落
        )
        snr_db = rx_power_dbm - float(station_cfg["noise_floor_dbm"]) + float(self.channel_cfg["snr_bias_db"])
        # 计算波达方向（DOA）
        doa_true = np.mod(np.degrees(np.arctan2(rel_vec[1], rel_vec[0])), 360.0)

        # 封装传播后的数据
        result = pulses.copy()
        result["scenario_id"] = self.scenario_cfg["scenario_id"]
        result["station_id"] = station_name
        result["receiver_mode"] = station_cfg["receiver_mode"]
        result["toa_true_station_us"] = result["toa_emission_us"].to_numpy(dtype=float) + delay_us
        result["rf_true_mhz"] = rf_values
        result["pw_true_us"] = result["pw_true_us"].to_numpy(dtype=float)
        result["pa_true_dbm"] = rx_power_dbm
        result["SNR_true_db"] = snr_db
        result["doa_true_deg"] = np.full(len(result), doa_true, dtype=float)
        return result

#模拟接收机时钟同步误差（真实基站无法完全时间同步)
class ReceiverClockModel:
    def __init__(self, seed: int) -> None:
        self.seed = int(seed)

    # 基站独立随机数
    def _station_rng(self, station_name: str) -> np.random.Generator:
        station_hash = sum(ord(char) for char in station_name)
        return np.random.default_rng(self.seed + station_hash * 809)

    # 注入时钟同步误差
    def apply(self, frame: pd.DataFrame, station_name: str, station_cfg: Dict[str, Any]) -> pd.DataFrame:
        if frame.empty:
            return frame
        result = frame.copy()
        toa_true = result["toa_true_station_us"].to_numpy(dtype=float)
        # 生成同步误差（均匀分布）
        sync_bound_us = float(station_cfg.get("time_sync_toa_error_us", 0.0))
        sync_rng = self._station_rng(station_name)
        sync_error_us = sync_rng.uniform(-sync_bound_us, sync_bound_us, size=len(result))
        # 带时钟误差的到达时间
        result["toa_clocked_us"] = toa_true + sync_error_us
        result["time_sync_error_us"] = result["toa_clocked_us"] - toa_true
        return result
    

class ReceiverDetectionModel:
    def __init__(self, scenario_cfg: Dict[str, Any], seed: int) -> None:
        self.scenario_cfg = scenario_cfg
        self.total_time_us = float(scenario_cfg["total_time_us"])
        self.seed = int(seed)
        self.station_burst_windows = self._build_burst_windows() # 突发丢失窗口

    # 基站随机数
    def _station_rng(self, station_name: str) -> np.random.Generator:
        station_hash = sum(ord(char) for char in station_name)
        return np.random.default_rng(self.seed + station_hash * 1_003)

    # 构建突发信号丢失窗口（模拟基站故障）
    def _build_burst_windows(self) -> Dict[str, List[Tuple[float, float]]]:
        # key：基站名字（如 "station1"）
        # value：该基站的故障时间窗口列表，每个窗口是 (开始时间, 结束时间)
        windows: Dict[str, List[Tuple[float, float]]] = {}
        for station_name, station_cfg in self.scenario_cfg["stations"].items():
            rng = self._station_rng(station_name)
            strength = float(station_cfg["burst_loss_strength"])
            # 计算丢失窗口数量：强度×12，四舍五入后限制在0~10之间
            n_windows = int(np.clip(np.rint(strength * 100), 0, 10))
            station_windows: List[Tuple[float, float]] = []
            for _ in range(n_windows):
                # 生成窗口持续时间：基础300~4500μs，强度越大，持续时间越长
                duration_us = float(rng.uniform(1000.0, 4500.0) * (1.0 + strength * 4))
                # 生成窗口开始时间：确保窗口不超出总时间范围
                start_us = float(rng.uniform(0.0, max(self.total_time_us - duration_us, 1.0)))
                station_windows.append((start_us, start_us + duration_us))
            windows[station_name] = station_windows
        return windows

    # 扫描模式盲区（扫描模式下，基站不在目标方向时无法接收）
    def _scan_blind_mask(self, frame: pd.DataFrame, station_cfg: Dict[str, Any]) -> np.ndarray:
        if frame.empty or station_cfg["receiver_mode"] != "scan":
            return np.zeros(len(frame), dtype=bool)
        cycle_us = max(float(station_cfg["scan_cycle_us"]), 1.0)
        dwell_us = float(station_cfg["scan_dwell_us"])
        phase_us = float(station_cfg["scan_phase_us"])
        phases = np.mod(frame["toa_clocked_us"].to_numpy(dtype=float) + phase_us, cycle_us)
        return phases > dwell_us

    def _burst_loss_mask(self, frame: pd.DataFrame, station_name: str, station_cfg: Dict[str, Any]) -> np.ndarray:
        # 处理空数据帧：如果没有脉冲数据，直接返回全False的掩码（无丢失）
        if frame.empty:
            return np.zeros(len(frame), dtype=bool)
        # 初始化突发丢失掩码：全False数组，True表示对应脉冲丢失
        burst_mask = np.zeros(len(frame), dtype=bool)
        # 提取所有脉冲的到达时间（TOA），单位：微秒（μs）
        toa_values = frame["toa_clocked_us"].to_numpy(dtype=float)
        # 从站点配置中读取“突发丢失强度”：控制丢失概率的大小
        strength = float(station_cfg["burst_loss_strength"])
        # 获取该站点专属的随机数生成器（种子包含站点名），保证丢失模式可复现
        rng = self._station_rng(f"{station_name}_burst")
        # 遍历该站点的所有“突发丢失时间窗口”
        for start_us, end_us in self.station_burst_windows.get(station_name, []):
            # 筛选出TOA落在当前丢失窗口内的脉冲（候选脉冲）
            candidate = (toa_values >= start_us) & (toa_values <= end_us)
            # 如果窗口内有候选脉冲，按概率标记丢失
            if np.any(candidate):
                # 计算丢失概率：基础概率0.6 + 强度系数，最高不超过0.98（避免100%丢失）
                loss_prob = min(0.98, 0.6 + strength)
                # 生成随机数，与丢失概率比较，得到随机丢失标记
                random_loss = rng.random(len(frame)) < loss_prob
                # 更新掩码：只有“在窗口内”且“随机命中丢失”的脉冲才标记为True（丢失）
                burst_mask |= candidate & random_loss
        # 返回最终的突发丢失掩码
        return burst_mask

    # 测量噪声+量化（模拟接收机硬件噪声）
    def _measurement_noise(self, frame: pd.DataFrame, station_cfg: Dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
        if frame.empty:
            return frame

        work = frame.copy()
        snr_linear = 10.0 ** (np.clip(work["SNR_true_db"].to_numpy(dtype=float), -20.0, 40.0) / 10.0)
        res_cfg = station_cfg["resolution"]
        noise_cfg = station_cfg["measurement_noise_floor"]

        # 按信噪比计算噪声标准差（SNR越低，噪声越大）
        toa_sigma = res_cfg["toa_us"] / 4.0 + 0.20 / np.sqrt(snr_linear + 1e-6)
        rf_sigma = noise_cfg["rf_mhz"] + 2.0 / np.sqrt(snr_linear + 1e-6)
        pw_sigma = noise_cfg["pw_us"] + 0.60 / np.sqrt(snr_linear + 1e-6)
        pa_sigma = noise_cfg["pa_db"] + 1.20 / np.sqrt(snr_linear + 1e-6)
        doa_sigma = noise_cfg["doa_deg"] + 9.0 / np.sqrt(snr_linear + 1e-6)

        # 注入高斯噪声
        toa_noisy = work["toa_clocked_us"].to_numpy(dtype=float) + rng.normal(0.0, toa_sigma)
        rf_noisy = work["rf_true_mhz"].to_numpy(dtype=float) + rng.normal(0.0, rf_sigma)
        pw_noisy = np.maximum(0.05, work["pw_true_us"].to_numpy(dtype=float) + rng.normal(0.0, pw_sigma))
        pa_noisy = work["pa_true_dbm"].to_numpy(dtype=float) + rng.normal(0.0, pa_sigma)
        doa_noisy = np.mod(work["doa_true_deg"].to_numpy(dtype=float) + rng.normal(0.0, doa_sigma), 360.0)

        # 量化处理
        toa_quantized, toa_bins = quantize_values(toa_noisy, res_cfg["toa_us"])
        rf_quantized, rf_bins = quantize_values(rf_noisy, res_cfg["rf_mhz"])
        pw_quantized, pw_bins = quantize_values(pw_noisy, res_cfg["pw_us"])
        pa_quantized, pa_bins = quantize_values(pa_noisy, res_cfg["pa_db"])

        # 保存带噪声、量化的数据
        work["TOA"] = toa_quantized
        work["RF"] = rf_quantized
        work["PW"] = pw_quantized
        work["PA"] = pa_quantized
        work["DOA"] = doa_noisy
        work["SNR"] = work["SNR_true_db"].to_numpy(dtype=float)
        work["resolution_bin_toa"] = toa_bins
        work["resolution_bin_rf"] = rf_bins
        work["resolution_bin_pw"] = pw_bins
        work["resolution_bin_pa"] = pa_bins
        return work

    # 脉冲重叠处理（重叠脉冲只保留SNR最高的）
    def _apply_overlap_resolution(self, work: pd.DataFrame, detected: np.ndarray, reasons: np.ndarray) -> None:
        candidate_indices = np.where(detected)[0]
        if len(candidate_indices) < 2:
            return
        sorted_candidates = candidate_indices[np.argsort(work.iloc[candidate_indices]["TOA"].to_numpy(dtype=float))]
        base_window_us = 0.25
        rf_tol_mhz = 12.0

        keep_idx = sorted_candidates[0]
        for current_idx in sorted_candidates[1:]:
            if not detected[keep_idx]:
                keep_idx = current_idx
                continue
            prev_toa = float(work.at[keep_idx, "TOA"])
            prev_pw = float(work.at[keep_idx, "PW"])
            curr_toa = float(work.at[current_idx, "TOA"])
            curr_pw = float(work.at[current_idx, "PW"])
            #时间接近：两个脉冲的 TOA 差 ≤ 时间窗口0.25us + 两者中较小脉宽的一半（考虑脉宽影响，若一个脉冲的上升沿落在另一个脉冲的脉宽内，更可能重叠）。
            #频率接近：两个脉冲的 RF 差 ≤ 频率容差12MHz（位于同一信道）。
            time_close = abs(curr_toa - prev_toa) <= base_window_us + 0.5 * min(prev_pw, curr_pw)
            rf_close = abs(float(work.at[current_idx, "RF"]) - float(work.at[keep_idx, "RF"])) <= rf_tol_mhz
            if not (time_close and rf_close): 
                keep_idx = current_idx
                continue

            # 重叠脉冲：丢弃SNR低的
            current_snr = float(work.at[current_idx, "SNR"])
            keep_snr = float(work.at[keep_idx, "SNR"])
            loser_idx = current_idx if current_snr < keep_snr else keep_idx
            detected[loser_idx] = False
            reasons[loser_idx] = "overlap_unresolved"
            if loser_idx == keep_idx:
                keep_idx = current_idx

    # 基站检测主函数
    def process_station(
        self,
        frame: pd.DataFrame,
        station_name: str,
        station_cfg: Dict[str, Any],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        rng = self._station_rng(f"{station_name}_detect")
        if frame.empty:
            # 返回空数据
            empty_obs = pd.DataFrame(columns=OBSERVATION_COLUMNS)
            empty_diag = pd.DataFrame(columns=["scenario_id","station_id","pulse_id","emitter_id","Label","toa_emission_us","toa_true_station_us","toa_observed_us","time_sync_error_us","SNR","is_detected","status"])
            return empty_obs, empty_diag, {"visible_truth_pulses": 0, "detected_pulses": 0}

        # 1. 注入测量噪声+量化
        work = self._measurement_noise(frame, station_cfg, rng)
        detected = np.ones(len(work), dtype=bool)
        reasons = np.full(len(work), "detected", dtype=object)

        # 2. 扫描盲区丢弃
        scan_blind = self._scan_blind_mask(work, station_cfg)
        detected[scan_blind] = False
        reasons[scan_blind] = "scan_blind"

        # 3. 突发丢失丢弃
        burst_loss = self._burst_loss_mask(work, station_name, station_cfg) & detected
        detected[burst_loss] = False
        reasons[burst_loss] = "burst_loss"

        # 4. 信噪比阈值+随机丢失
        remaining = detected.copy()
        snr = work["SNR"].to_numpy(dtype=float)
        threshold = float(station_cfg["threshold_snr_db"])
        receiver_miss = float(station_cfg["receiver_miss_probability"])

        # 根据 SNR 与检测门限的差值 计算一个检测概率
        pd_curve = 1.0 / (1.0 + np.exp(-(snr - threshold) / 2.2))
        # 进一步乘以一个基于 receiver_miss 的衰减因子，模拟接收机在边缘条件下的额外丢失概率
        # receiver_miss_probability 会整体压低检测概率，尤其是在 SNR 接近门限时更明显。这模拟了接收机在实际操作中可能出现的性能不稳定性。
        pd_curve *= max(0.0, 1.0 - receiver_miss)

        # 如果随机数 random_draw 大于等于检测概率 pd_curve，说明这个脉冲没有被检测到。
        # 如果这个漏检脉冲的 SNR 很低，即 snr < threshold + 0.5，则原因记为 below_threshold。
        # 否则，即 SNR 不算特别低，但仍然没检测到，则原因记为 receiver_miss。
        random_draw = rng.random(len(work))
        miss_mask = remaining & (random_draw >= pd_curve)

        below_threshold = miss_mask & (snr < threshold + 0.5)
        detected[below_threshold] = False
        reasons[below_threshold] = "below_threshold"
        receiver_miss_mask = miss_mask & (~below_threshold)
        detected[receiver_miss_mask] = False
        reasons[receiver_miss_mask] = "receiver_miss"

        # 5. 脉冲重叠处理
        self._apply_overlap_resolution(work, detected, reasons)
        work["is_detected"] = detected
        work["status"] = reasons

        # 筛选检测到的脉冲+诊断数据
        observed = work.loc[detected, OBSERVATION_COLUMNS].copy()
        diagnostics = work[["scenario_id","station_id","pulse_id","emitter_id","Label","toa_emission_us","toa_true_station_us","TOA","time_sync_error_us","SNR","is_detected","status"]].copy()
        diagnostics.rename(columns={"TOA": "toa_observed_us"}, inplace=True)

        # 统计汇总
        summary = {
            "visible_truth_pulses": int(len(work)),
            "detected_pulses": int(detected.sum()),
            "missed_pulses": int((~detected).sum()),
            "average_snr_db": float(work["SNR"].mean()) if len(work) else float("nan"),
            "average_detected_snr_db": float(work.loc[detected, "SNR"].mean()) if detected.any() else float("nan"),
            "time_sync_toa_error_ns": float(station_cfg.get("time_sync_toa_error_ns", 0.0)),
            "time_sync_toa_error_mean_us": float(work["time_sync_error_us"].mean()),
            "time_sync_toa_error_max_us": float(np.max(np.abs(work["time_sync_error_us"]))),
            "reason_counts": pd.Series(reasons).value_counts().to_dict(),
        }
        return observed, diagnostics, summary
    

class ObservationWriter:
    def __init__(self, scenario_cfg: Dict[str, Any], output_dir: Path, seed: int) -> None:
        self.scenario_cfg = scenario_cfg
        self.output_dir = output_dir
        self.seed = int(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _station_rng(self, station_name: str) -> np.random.Generator:
        station_hash = sum(ord(char) for char in station_name)
        return np.random.default_rng(self.seed + station_hash * 503)

    def _interference_count(self, reference_count: int) -> int:
        ratio = float(self.scenario_cfg["interference"]["structured_interference_ratio"])
        return max(1, int(reference_count * ratio))

    def _quantized_interference(
        self,
        raw_df: pd.DataFrame,
        station_name: str,
        station_cfg: Dict[str, Any],
    ) -> pd.DataFrame:
        if raw_df.empty:
            return raw_df
        res_cfg = station_cfg["resolution"]
        raw_df = raw_df.copy()
        raw_df["TOA"], raw_df["resolution_bin_toa"] = quantize_values(raw_df["TOA"].to_numpy(dtype=float), res_cfg["toa_us"])
        raw_df["RF"], raw_df["resolution_bin_rf"] = quantize_values(raw_df["RF"].to_numpy(dtype=float), res_cfg["rf_mhz"])
        raw_df["PW"], raw_df["resolution_bin_pw"] = quantize_values(raw_df["PW"].to_numpy(dtype=float), res_cfg["pw_us"])
        raw_df["PA"], raw_df["resolution_bin_pa"] = quantize_values(raw_df["PA"].to_numpy(dtype=float), res_cfg["pa_db"])
        raw_df["DOA"] = np.mod(raw_df["DOA"].to_numpy(dtype=float), 360.0)
        raw_df["station_id"] = station_name
        raw_df["scenario_id"] = self.scenario_cfg["scenario_id"]
        raw_df["receiver_mode"] = station_cfg["receiver_mode"]
        raw_df["emitter_id"] = -1
        raw_df["Label"] = -1
        return raw_df[OBSERVATION_COLUMNS]

    def generate_structured_interference(
        self,
        station_name: str,
        station_cfg: Dict[str, Any],
        detected_df: pd.DataFrame,
        reference_truth_count: int,
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        rng = self._station_rng(station_name)
        count = self._interference_count(max(reference_truth_count, len(detected_df), 16))
        shares = self.scenario_cfg["interference"]
        counts = {
            "thermal_noise": max(1, int(count * shares["thermal_share"])),
            "cochannel_noise": max(1, int(count * shares["cochannel_share"])),
            "cluster_false_target": max(1, int(count * shares["cluster_share"])),
        }
        counts["decoy_pulse"] = max(
            1,
            count - counts["thermal_noise"] - counts["cochannel_noise"] - counts["cluster_false_target"],
        )

        total_time_us = float(self.scenario_cfg["total_time_us"])
        noise_floor = float(station_cfg["noise_floor_dbm"])
        rows: List[pd.DataFrame] = []

        thermal = pd.DataFrame(
            {
                "TOA": rng.uniform(0.0, total_time_us, counts["thermal_noise"]),
                "RF": rng.uniform(2_000.0, 10_000.0, counts["thermal_noise"]),
                "PW": rng.uniform(0.1, 6.0, counts["thermal_noise"]),
                "PA": rng.uniform(noise_floor - 6.0, noise_floor + 2.0, counts["thermal_noise"]),
                "DOA": rng.uniform(0.0, 360.0, counts["thermal_noise"]),
                "SNR": rng.uniform(-10.0, 2.0, counts["thermal_noise"]),
            }
        )
        rows.append(self._quantized_interference(thermal, station_name, station_cfg))

        if detected_df.empty:
            rf_reference = rng.uniform(7_000.0, 9_500.0, counts["cochannel_noise"])
            pw_reference = rng.uniform(8.0, 25.0, counts["cochannel_noise"])
            doa_reference = rng.uniform(0.0, 360.0, counts["cochannel_noise"])
        else:
            sampled = detected_df.sample(n=counts["cochannel_noise"], replace=True, random_state=int(rng.integers(1, 10_000)))
            rf_reference = sampled["RF"].to_numpy(dtype=float)
            pw_reference = sampled["PW"].to_numpy(dtype=float)
            doa_reference = sampled["DOA"].to_numpy(dtype=float)

        cochannel = pd.DataFrame(
            {
                "TOA": rng.uniform(0.0, total_time_us, counts["cochannel_noise"]),
                "RF": rf_reference + rng.normal(0.0, 3.0, counts["cochannel_noise"]),
                "PW": np.maximum(0.1, pw_reference + rng.normal(0.0, 0.4, counts["cochannel_noise"])),
                "PA": rng.uniform(noise_floor + 2.0, noise_floor + 10.0, counts["cochannel_noise"]),
                "DOA": doa_reference + rng.normal(0.0, 6.0, counts["cochannel_noise"]),
                "SNR": rng.uniform(-2.0, 8.0, counts["cochannel_noise"]),
            }
        )
        rows.append(self._quantized_interference(cochannel, station_name, station_cfg))

        cluster_centers = rng.uniform(0.0, total_time_us, max(1, counts["cluster_false_target"] // 6))
        repeat_count = int(np.ceil(counts["cluster_false_target"] / len(cluster_centers)))
        cluster_toa = np.repeat(cluster_centers, repeat_count)[: counts["cluster_false_target"]]
        cluster = pd.DataFrame(
            {
                "TOA": cluster_toa + rng.normal(0.0, 4.0, counts["cluster_false_target"]),
                "RF": rng.uniform(7_400.0, 9_600.0, counts["cluster_false_target"]),
                "PW": rng.uniform(10.0, 28.0, counts["cluster_false_target"]),
                "PA": rng.uniform(noise_floor + 3.0, noise_floor + 12.0, counts["cluster_false_target"]),
                "DOA": rng.uniform(0.0, 360.0, counts["cluster_false_target"]),
                "SNR": rng.uniform(0.0, 10.0, counts["cluster_false_target"]),
            }
        )
        rows.append(self._quantized_interference(cluster, station_name, station_cfg))

        if detected_df.empty:
            decoy_base = pd.DataFrame(
                {
                    "TOA": rng.uniform(0.0, total_time_us, counts["decoy_pulse"]),
                    "RF": rng.uniform(7_500.0, 9_500.0, counts["decoy_pulse"]),
                    "PW": rng.uniform(10.0, 35.0, counts["decoy_pulse"]),
                    "PA": rng.uniform(noise_floor + 4.0, noise_floor + 14.0, counts["decoy_pulse"]),
                    "DOA": rng.uniform(0.0, 360.0, counts["decoy_pulse"]),
                    "SNR": rng.uniform(2.0, 12.0, counts["decoy_pulse"]),
                }
            )
        else:
            sampled = detected_df.sample(n=counts["decoy_pulse"], replace=True, random_state=int(rng.integers(1, 10_000)))
            decoy_base = pd.DataFrame(
                {
                    "TOA": sampled["TOA"].to_numpy(dtype=float) + rng.normal(0.0, 0.25, counts["decoy_pulse"]),
                    "RF": sampled["RF"].to_numpy(dtype=float) + rng.normal(0.0, 1.2, counts["decoy_pulse"]),
                    "PW": np.maximum(0.1, sampled["PW"].to_numpy(dtype=float) + rng.normal(0.0, 0.15, counts["decoy_pulse"])),
                    "PA": sampled["PA"].to_numpy(dtype=float) + rng.normal(-1.0, 0.6, counts["decoy_pulse"]),
                    "DOA": sampled["DOA"].to_numpy(dtype=float) + rng.normal(0.0, 2.0, counts["decoy_pulse"]),
                    "SNR": sampled["SNR"].to_numpy(dtype=float) + rng.normal(-1.0, 1.0, counts["decoy_pulse"]),
                }
            )
        rows.append(self._quantized_interference(decoy_base, station_name, station_cfg))

        interference = pd.concat(rows, ignore_index=True).sort_values("TOA").reset_index(drop=True)
        return interference, counts

    def write_station_csv(
        self,
        station_name: str,
        observed_df: pd.DataFrame,
        interference_df: pd.DataFrame,
    ) -> Path:
        combined = pd.concat([observed_df, interference_df], ignore_index=True)
        if combined.empty:
            combined = pd.DataFrame(columns=OBSERVATION_COLUMNS)
        combined = combined.sort_values("TOA").reset_index(drop=True)
        save_path = self.output_dir / f"{station_name}_data.csv"
        combined.to_csv(save_path, index=False)
        return save_path

    def write_per_station_summary(self, rows: List[Dict[str, Any]]) -> Path:
        summary_path = self.output_dir / "per_station_summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        return summary_path

    def write_pulse_mapping(self, diagnostics: pd.DataFrame) -> Path:
        mapping_path = self.output_dir / "pulse_mapping.csv"
        if diagnostics.empty:
            pd.DataFrame().to_csv(mapping_path, index=False)
            return mapping_path

        station_names = list(self.scenario_cfg["stations"].keys())
        rows: List[Dict[str, Any]] = []
        for pulse_id, group in diagnostics.groupby("pulse_id", sort=False):
            first = group.iloc[0]
            row: Dict[str, Any] = {
                "scenario_id": first["scenario_id"],
                "pulse_id": pulse_id,
                "emitter_id": int(first["emitter_id"]),
                "Label": int(first["Label"]),
                "toa_emission_us": float(first["toa_emission_us"]),
                "detected_station_count": int(group["is_detected"].sum()),
            }
            for station_name in station_names:
                station_rows = group[group["station_id"] == station_name]
                if station_rows.empty:
                    row[f"{station_name}_status"] = "not_visible"
                    row[f"{station_name}_toa_true_us"] = np.nan
                    row[f"{station_name}_toa_observed_us"] = np.nan
                else:
                    station_row = station_rows.iloc[0]
                    row[f"{station_name}_status"] = station_row["status"]
                    row[f"{station_name}_toa_true_us"] = station_row["toa_true_station_us"]
                    row[f"{station_name}_toa_observed_us"] = station_row["toa_observed_us"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(mapping_path, index=False)
        return mapping_path

    def write_manifest(self, manifest: Dict[str, Any]) -> Path:
        manifest_path = self.output_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file_obj:
            json.dump(to_serializable(manifest), file_obj, indent=2, ensure_ascii=False)
        return manifest_path

class PDWScenarioSimulator:
    def __init__(self, scenario_cfg: Dict[str, Any]) -> None:
        self.scenario_cfg = scenario_cfg
        self.output_dir = Path(scenario_cfg["output_dir"])
        self.seed = int(scenario_cfg["seed"])
        self.emitter_model = EmitterModel(
            scenario_cfg["emitters"],
            scenario_cfg["total_time_us"],
            seed=self.seed,
        )
        self.channel_model = ChannelModel(scenario_cfg, seed=self.seed + 17)
        self.clock_model = ReceiverClockModel(seed=self.seed + 23)
        self.detection_model = ReceiverDetectionModel(scenario_cfg, seed=self.seed + 29)
        self.writer = ObservationWriter(scenario_cfg, self.output_dir, seed=self.seed + 41)

    def _build_not_visible_diagnostics(
        self,
        pulses: pd.DataFrame,
        station_name: str,
    ) -> pd.DataFrame:
        if pulses.empty:
            return pd.DataFrame()
        return pd.DataFrame(
            {
                "scenario_id": self.scenario_cfg["scenario_id"],
                "station_id": station_name,
                "pulse_id": pulses["pulse_id"].to_numpy(),
                "emitter_id": pulses["emitter_id"].to_numpy(dtype=int),
                "Label": pulses["Label"].to_numpy(dtype=int),
                "toa_emission_us": pulses["toa_emission_us"].to_numpy(dtype=float),
                "toa_true_station_us": np.full(len(pulses), np.nan),
                "toa_observed_us": np.full(len(pulses), np.nan),
                "time_sync_error_us": np.full(len(pulses), np.nan),
                "SNR": np.full(len(pulses), np.nan),
                "is_detected": np.zeros(len(pulses), dtype=bool),
                "status": np.full(len(pulses), "not_visible", dtype=object),
            }
        )

    def run(self) -> Dict[str, Any]:
        # 1. 初始化输出目录并生成发射端真值脉冲序列。
        # 2. 逐站完成传播、时钟误差注入、检测判决和结构化干扰注入。
        # 3. 写出观测 CSV、站点汇总、脉冲映射和 manifest。
        self.output_dir.mkdir(parents=True, exist_ok=True)
        pulses, emitter_stats = self.emitter_model.generate()
        logging.info(
            "Scenario %s started with %d emitted pulses.",
            self.scenario_cfg["scenario_id"],
            len(pulses),
        )

        all_diagnostics: List[pd.DataFrame] = []
        per_station_rows: List[Dict[str, Any]] = []
        output_files: Dict[str, str] = {}
        interference_summary: Dict[str, Dict[str, int]] = {}

        emitter_lookup = {int(emitter["id"]): emitter for emitter in self.scenario_cfg["emitters"]}
        station_ids = list(self.scenario_cfg["stations"].keys())

        for station_name in station_ids:
            station_cfg = self.scenario_cfg["stations"][station_name]
            station_truth_frames: List[pd.DataFrame] = []
            station_invisible_frames: List[pd.DataFrame] = []

            for emitter_id, emitter_cfg in emitter_lookup.items():
                emitter_pulses = pulses[pulses["emitter_id"] == emitter_id].copy()
                if emitter_pulses.empty:
                    continue
                if station_name in emitter_cfg.get("visible_stations", []):
                    propagated = self.channel_model.propagate(
                        emitter_pulses,
                        emitter_cfg,
                        station_name,
                        station_cfg,
                    )
                    if not propagated.empty:
                        station_truth_frames.append(propagated)
                else:
                    station_invisible_frames.append(self._build_not_visible_diagnostics(emitter_pulses, station_name))

            station_truth = (
                pd.concat(station_truth_frames, ignore_index=True)
                if station_truth_frames
                else pd.DataFrame()
            )
            if not station_truth.empty:
                station_truth = self.clock_model.apply(station_truth, station_name, station_cfg)
                station_truth.sort_values("toa_true_station_us", inplace=True)
                station_truth.reset_index(drop=True, inplace=True)

            observed_df, diagnostics_df, station_summary = self.detection_model.process_station(
                station_truth,
                station_name,
                station_cfg,
            )
            if station_invisible_frames:
                diagnostics_df = pd.concat([diagnostics_df] + station_invisible_frames, ignore_index=True)
            diagnostics_df.sort_values(["pulse_id", "station_id"], inplace=True)
            all_diagnostics.append(diagnostics_df)

            interference_df, interference_stats = self.writer.generate_structured_interference(
                station_name,
                station_cfg,
                observed_df,
                station_summary.get("visible_truth_pulses", 0),
            )
            interference_summary[station_name] = interference_stats
            csv_path = self.writer.write_station_csv(station_name, observed_df, interference_df)
            output_files[f"{station_name}_csv"] = str(csv_path)

            station_row = {
                "scenario_id": self.scenario_cfg["scenario_id"],
                "station_id": station_name,
                "receiver_mode": station_cfg["receiver_mode"],
                "time_sync_toa_error_ns": station_summary.get("time_sync_toa_error_ns", np.nan),
                "visible_truth_pulses": station_summary.get("visible_truth_pulses", 0),
                "detected_pulses": station_summary.get("detected_pulses", 0),
                "missed_pulses": station_summary.get("missed_pulses", 0),
                "detection_rate": (
                    station_summary["detected_pulses"] / station_summary["visible_truth_pulses"]
                    if station_summary.get("visible_truth_pulses", 0)
                    else 0.0
                ),
                "average_snr_db": station_summary.get("average_snr_db", np.nan),
                "average_detected_snr_db": station_summary.get("average_detected_snr_db", np.nan),
                "time_sync_toa_error_mean_us": station_summary.get("time_sync_toa_error_mean_us", np.nan),
                "time_sync_toa_error_max_us": station_summary.get("time_sync_toa_error_max_us", np.nan),
                "interference_total": int(sum(interference_stats.values())),
            }
            for reason, count in station_summary.get("reason_counts", {}).items():
                station_row[f"reason_{reason}"] = int(count)
            for category, count in interference_stats.items():
                station_row[f"interference_{category}"] = int(count)
            per_station_rows.append(station_row)

            logging.info(
                "Station %s finished: %d detected PDWs, %d interference pulses.",
                station_name,
                len(observed_df),
                len(interference_df),
            )

        diagnostics = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
        station_summary_path = self.writer.write_per_station_summary(per_station_rows)
        output_files["per_station_summary_csv"] = str(station_summary_path)
        pulse_mapping_path = self.writer.write_pulse_mapping(diagnostics)
        output_files["pulse_mapping_csv"] = str(pulse_mapping_path)

        manifest = {
            "scenario_id": self.scenario_cfg["scenario_id"],
            "difficulty": self.scenario_cfg["difficulty"],
            "seed": self.scenario_cfg["seed"],
            "total_time_us": self.scenario_cfg["total_time_us"],
            "axes": self.scenario_cfg["axes"],
            "global_channel": self.scenario_cfg["global_channel"],
            "stations": self.scenario_cfg["stations"],
            "emitters": self.scenario_cfg["emitters"],
            "emitter_stats": emitter_stats,
            "per_station_summary": per_station_rows,
            "interference_summary": interference_summary,
            "totals": {
                "emitted_pulses": int(len(pulses)),
                "detected_pulses": int(diagnostics["is_detected"].sum()) if not diagnostics.empty else 0,
                "undetected_pulses": int((~diagnostics["is_detected"]).sum()) if not diagnostics.empty else 0,
            },
            "output_files": output_files,
        }
        manifest_path = self.writer.write_manifest(manifest)
        output_files["manifest_json"] = str(manifest_path)
        manifest["output_files"] = output_files
        self.writer.write_manifest(manifest)
        logging.info("Scenario %s completed.", self.scenario_cfg["scenario_id"])
        return manifest


def main() -> Dict[str, Any]:
    # 主入口：构造默认场景配置，并执行完整的多站 PDW 仿真流程。
    config = get_default_simulation_config(
        difficulty="hard",
        seed=42,
        output_dir="./datasets/generated_data",
        total_time_us=DEFAULT_TOTAL_TIME_US,
    )
    simulator = PDWScenarioSimulator(config)
    return simulator.run()


if __name__ == "__main__":
    main()

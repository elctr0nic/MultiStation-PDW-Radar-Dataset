from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    from pdw_sim_config import get_station_coords
except ImportError:
    from datasets.pdw_sim_config import get_station_coords


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
    datefmt="%H:%M:%S",
)


C_LIGHT_KM_S = 3e5
TRAIN_DIR = Path(r"E:\PyCharm\Py_Projects\mulstation_learning\训练数据集")
OUTPUT_DIR = Path("./datasets/generated_data")
OBSERVATION_COLUMNS = [
    "TOA",
    "RF",
    "PW",
    "PA",
    "DOA",
    "Label",
    "SNR",
    "station_id",
    "scenario_id",
    "receiver_mode",
    "emitter_id",
    "resolution_bin_toa",
    "resolution_bin_rf",
    "resolution_bin_pw",
    "resolution_bin_pa",
]


DEFAULT_STATION_COORDS = get_station_coords()
CONVERTER_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "scenario_id": "ebdsc_multistation_seed42",
    "source_train_dir": str(TRAIN_DIR),
    "output_dir": str(OUTPUT_DIR),
    "num_emitters": 5,
    "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
    "selected_output_split": "train",
    "stations": {
        "S1": {
            "coord": DEFAULT_STATION_COORDS["S1"].tolist(),
            "receiver_mode": "stare",
            "snr_db_mean": 14.0,
            "miss_ratio": 0.05,
            "spurious_ratio": 0.08,
            "time_sync_error_ns": 20.0,
            "scan_cycle_us": 0.0,
            "scan_dwell_us": 0.0,
            "scan_phase_us": 0.0,
        },
        "S2": {
            "coord": DEFAULT_STATION_COORDS["S2"].tolist(),
            "receiver_mode": "scan",
            "snr_db_mean": 10.0,
            "miss_ratio": 0.10,
            "spurious_ratio": 0.12,
            "time_sync_error_ns": 35.0,
            "scan_cycle_us": 1_200.0,
            "scan_dwell_us": 720.0,
            "scan_phase_us": 140.0,
        },
        "S3": {
            "coord": DEFAULT_STATION_COORDS["S3"].tolist(),
            "receiver_mode": "stare",
            "snr_db_mean": 16.0,
            "miss_ratio": 0.03,
            "spurious_ratio": 0.06,
            "time_sync_error_ns": 25.0,
            "scan_cycle_us": 0.0,
            "scan_dwell_us": 0.0,
            "scan_phase_us": 0.0,
        },
    },
    "emitters": [
        {"enabled": True, "source_label": 1, "slice_duration_s": 2.5, "position": [120.0, 80.0, 20.0]},
        {"enabled": True, "source_label": 2, "slice_duration_s": 3.0, "position": [150.0, 30.0, 60.0]},
        {"enabled": True, "source_label": 3, "slice_duration_s": 2.0, "position": [90.0, 170.0, 30.0]},
        {"enabled": True, "source_label": 4, "slice_duration_s": 2.8, "position": [180.0, 110.0, 45.0]},
        {"enabled": True, "source_label": 5, "slice_duration_s": 3.2, "position": [60.0, 210.0, 55.0]},
        {"enabled": True, "source_label": 6, "slice_duration_s": 2.2, "position": [210.0, 70.0, 35.0]},
        {"enabled": True, "source_label": 7, "slice_duration_s": 2.6, "position": [80.0, 260.0, 40.0]},
        {"enabled": True, "source_label": 8, "slice_duration_s": 3.4, "position": [240.0, 140.0, 70.0]},
        {"enabled": True, "source_label": 9, "slice_duration_s": 2.4, "position": [140.0, 240.0, 50.0]},
        {"enabled": True, "source_label": 10, "slice_duration_s": 2.1, "position": [260.0, 50.0, 25.0]},
        {"enabled": True, "source_label": 11, "slice_duration_s": 2.7, "position": [200.0, 200.0, 80.0]},
        {"enabled": True, "source_label": 12, "slice_duration_s": 1.8, "position": [100.0, 300.0, 20.0]},
    ],
}


def to_serializable(value: Any) -> Any:
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


def parse_label_from_filename(file_path: Path) -> int:
    match = re.search(r"信号类型(\d+)训练集", file_path.stem)
    if not match:
        raise ValueError(f"无法从文件名中解析标签: {file_path}")
    return int(match.group(1))


def load_clean_emitter_files(source_train_dir: Path) -> Dict[int, Dict[str, Any]]:
    """读取 12 个纯净训练文件，返回按标签索引的数据表。"""
    if not source_train_dir.exists():
        raise FileNotFoundError(f"训练数据目录不存在: {source_train_dir}")

    files = sorted(source_train_dir.glob("信号类型*训练集.txt"), key=parse_label_from_filename)
    if len(files) != 12:
        raise ValueError(f"预期找到 12 个训练文件，实际找到 {len(files)} 个。")

    dataset_map: Dict[int, Dict[str, Any]] = {}
    for file_path in files:
        label = parse_label_from_filename(file_path)
        frame = pd.read_csv(
            file_path,
            sep=r"\s+",
            names=["TOA_s", "RF", "PW", "PA", "DOA", "Label"],
            engine="python",
        )
        frame = frame.astype(
            {
                "TOA_s": np.float64,
                "RF": np.float64,
                "PW": np.float64,
                "PA": np.float64,
                "DOA": np.float64,
                "Label": np.int64,
            }
        ).sort_values("TOA_s", ignore_index=True)
        unique_labels = frame["Label"].unique()
        if len(unique_labels) != 1 or int(unique_labels[0]) != label:
            raise ValueError(f"{file_path.name} 的标签列与文件名标签不一致。")

        dataset_map[label] = {
            "label": label,
            "file_path": file_path,
            "data": frame,
            "duration_s": float(frame["TOA_s"].iloc[-1] - frame["TOA_s"].iloc[0]),
        }

    return dataset_map


def select_emitters(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从配置中选出本次场景启用的辐射源。"""
    enabled_emitters = [item for item in config["emitters"] if item.get("enabled", False)]
    selected = enabled_emitters[: int(config["num_emitters"])]
    if len(selected) < int(config["num_emitters"]):
        raise ValueError("启用的辐射源数量少于 num_emitters。")
    return selected


def build_emitter_truth_slice(
    emitter_cfg: Dict[str, Any],
    source_entry: Dict[str, Any],
    rng: np.random.Generator,
    emitter_id: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """从单类纯净源中随机截取一段时间，构造该辐射源的参考真值流。"""
    source_df = source_entry["data"]
    source_toa = source_df["TOA_s"].to_numpy(dtype=float)
    slice_duration_s = float(emitter_cfg["slice_duration_s"])
    source_start = float(source_toa[0])
    source_end = float(source_toa[-1])
    available_duration_s = max(0.0, source_end - source_start)

    if slice_duration_s >= available_duration_s:
        slice_start_s = source_start
        slice_end_s = source_end
        sliced = source_df.copy()
    else:
        max_start = source_end - slice_duration_s
        slice_start_s = float(rng.uniform(source_start, max_start))
        slice_end_s = slice_start_s + slice_duration_s
        sliced = source_df[(source_df["TOA_s"] >= slice_start_s) & (source_df["TOA_s"] < slice_end_s)].copy()
        if sliced.empty:
            nearest_idx = int(np.searchsorted(source_toa, slice_start_s))
            nearest_idx = min(max(nearest_idx, 0), len(source_df) - 1)
            sliced = source_df.iloc[[nearest_idx]].copy()
            slice_start_s = float(sliced["TOA_s"].iloc[0])
            slice_end_s = min(source_end, slice_start_s + slice_duration_s)

    sliced.reset_index(drop=True, inplace=True)
    sliced["TOA_s"] = sliced["TOA_s"] - float(sliced["TOA_s"].iloc[0])
    sliced["toa_true_us"] = sliced["TOA_s"] * 1e6
    sliced["pulse_index"] = np.arange(len(sliced), dtype=np.int64)
    sliced["pulse_id"] = [f"E{emitter_id:02d}_P{idx:07d}" for idx in sliced["pulse_index"]]
    sliced["emitter_id"] = emitter_id
    sliced["source_label"] = int(source_entry["label"])
    sliced["source_file"] = source_entry["file_path"].name
    sliced["position_x"] = float(emitter_cfg["position"][0])
    sliced["position_y"] = float(emitter_cfg["position"][1])
    sliced["position_z"] = float(emitter_cfg["position"][2])
    sliced["rf_true_mhz"] = sliced["RF"].to_numpy(dtype=float)
    sliced["pw_true_us"] = sliced["PW"].to_numpy(dtype=float)
    sliced["pa_true_db"] = sliced["PA"].to_numpy(dtype=float)
    sliced["doa_true_deg"] = sliced["DOA"].to_numpy(dtype=float)
    sliced["Label"] = sliced["Label"].to_numpy(dtype=int)

    truth_df = sliced[
        [
            "pulse_id",
            "pulse_index",
            "emitter_id",
            "source_label",
            "source_file",
            "Label",
            "toa_true_us",
            "rf_true_mhz",
            "pw_true_us",
            "pa_true_db",
            "doa_true_deg",
            "position_x",
            "position_y",
            "position_z",
        ]
    ].copy()

    meta = {
        "emitter_id": emitter_id,
        "source_label": int(source_entry["label"]),
        "source_file": source_entry["file_path"].name,
        "slice_duration_s": slice_duration_s,
        "slice_start_s": slice_start_s,
        "slice_end_s": slice_end_s,
        "position": [float(v) for v in emitter_cfg["position"]],
        "truth_pulse_count": int(len(truth_df)),
    }
    return truth_df, meta


def merge_truth_emitters(truth_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """合并所有辐射源真值流，并按全局 TOA 排序。"""
    if not truth_frames:
        return pd.DataFrame()
    merged = pd.concat(truth_frames, ignore_index=True)
    merged.sort_values("toa_true_us", inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def split_truth_stream(
    truth_df: pd.DataFrame, split_ratios: Dict[str, float]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, float]]]:
    """按全局时间轴切分 train/val/test。"""
    if truth_df.empty:
        return {}, {}

    ratios = dict(split_ratios)
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("split_ratios 的总和必须大于 0。")
    ratios = {key: value / total_ratio for key, value in ratios.items()}

    max_toa_us = float(truth_df["toa_true_us"].max())
    train_end = ratios["train"] * max_toa_us
    val_end = (ratios["train"] + ratios["val"]) * max_toa_us
    split_ranges = {
        "train": (0.0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, np.nextafter(max_toa_us, np.inf)),
    }

    split_frames: Dict[str, pd.DataFrame] = {}
    split_meta: Dict[str, Dict[str, float]] = {}
    for split_name, (start_us, end_us) in split_ranges.items():
        mask = (truth_df["toa_true_us"] >= start_us) & (truth_df["toa_true_us"] < end_us)
        split_df = truth_df.loc[mask].copy().reset_index(drop=True)
        if not split_df.empty:
            split_df["toa_true_us"] = split_df["toa_true_us"] - float(split_df["toa_true_us"].iloc[0])
        split_frames[split_name] = split_df
        split_meta[split_name] = {
            "start_us": float(start_us),
            "end_us": float(end_us),
            "pulse_count": int(len(split_df)),
        }

    return split_frames, split_meta


def compute_propagation_delay_us(emitter_positions: np.ndarray, station_coord: np.ndarray) -> np.ndarray:
    """根据辐射源位置和站点位置计算传播时延。"""
    distances_km = np.linalg.norm(emitter_positions - station_coord[None, :], axis=1)
    return (distances_km / C_LIGHT_KM_S) * 1e6


def build_scan_blind_mask(toa_values_us: np.ndarray, station_cfg: Dict[str, Any]) -> np.ndarray:
    """扫描模式下，非驻留时段视为不可见。"""
    if station_cfg["receiver_mode"] != "scan":
        return np.zeros(len(toa_values_us), dtype=bool)
    cycle_us = max(float(station_cfg["scan_cycle_us"]), 1.0)
    dwell_us = float(station_cfg["scan_dwell_us"])
    phase_us = float(station_cfg["scan_phase_us"])
    phase = np.mod(toa_values_us + phase_us, cycle_us)
    return phase > dwell_us


def project_truth_to_station(
    truth_df: pd.DataFrame,
    station_name: str,
    station_cfg: Dict[str, Any],
    scenario_id: str,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """将参考真值流投影到单站观测。"""
    if truth_df.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS), {
            "detected_truth_pulses": 0,
            "scan_dropped_pulses": 0,
            "random_missed_pulses": 0,
        }

    station_coord = np.asarray(station_cfg["coord"], dtype=float)
    emitter_positions = truth_df[["position_x", "position_y", "position_z"]].to_numpy(dtype=float)
    propagation_delay_us = compute_propagation_delay_us(emitter_positions, station_coord)
    sync_error_bound_us = float(station_cfg["time_sync_error_ns"]) * 1e-3
    sync_error_us = rng.uniform(-sync_error_bound_us, sync_error_bound_us, size=len(truth_df))

    toa_station_us = truth_df["toa_true_us"].to_numpy(dtype=float) + propagation_delay_us + sync_error_us
    scan_blind_mask = build_scan_blind_mask(toa_station_us, station_cfg)
    miss_mask = rng.random(len(truth_df)) < float(station_cfg["miss_ratio"])
    detected_mask = ~(scan_blind_mask | miss_mask)

    observed = truth_df.loc[detected_mask].copy().reset_index(drop=True)
    if observed.empty:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS), {
            "detected_truth_pulses": 0,
            "scan_dropped_pulses": int(scan_blind_mask.sum()),
            "random_missed_pulses": int((~scan_blind_mask & miss_mask).sum()),
        }

    observed_idx = np.where(detected_mask)[0]
    snr_values = float(station_cfg["snr_db_mean"]) + rng.normal(0.0, 1.5, size=len(observed))
    rf_values = observed["rf_true_mhz"].to_numpy(dtype=float) + rng.normal(0.0, 1.0, size=len(observed))
    pw_noise = np.maximum(0.02, 0.015 * observed["pw_true_us"].to_numpy(dtype=float))
    pw_values = np.maximum(
        0.05,
        observed["pw_true_us"].to_numpy(dtype=float) + rng.normal(0.0, pw_noise, size=len(observed)),
    )
    pa_station_bias = (float(station_cfg["snr_db_mean"]) - 12.0) * 0.2
    pa_values = observed["pa_true_db"].to_numpy(dtype=float) + pa_station_bias + rng.normal(0.0, 0.8, size=len(observed))
    doa_values = np.mod(
        observed["doa_true_deg"].to_numpy(dtype=float) + rng.normal(0.0, 1.5, size=len(observed)),
        360.0,
    )

    station_frame = pd.DataFrame(
        {
            "TOA": toa_station_us[observed_idx],
            "RF": rf_values,
            "PW": pw_values,
            "PA": pa_values,
            "DOA": doa_values,
            "Label": observed["Label"].to_numpy(dtype=int),
            "SNR": snr_values,
            "station_id": station_name,
            "scenario_id": scenario_id,
            "receiver_mode": station_cfg["receiver_mode"],
            "emitter_id": observed["emitter_id"].to_numpy(dtype=int),
            "resolution_bin_toa": np.full(len(observed), -1, dtype=int),
            "resolution_bin_rf": np.full(len(observed), -1, dtype=int),
            "resolution_bin_pw": np.full(len(observed), -1, dtype=int),
            "resolution_bin_pa": np.full(len(observed), -1, dtype=int),
        }
    )
    station_frame.sort_values("TOA", inplace=True)
    station_frame.reset_index(drop=True, inplace=True)
    return station_frame, {
        "detected_truth_pulses": int(len(station_frame)),
        "scan_dropped_pulses": int(scan_blind_mask.sum()),
        "random_missed_pulses": int((~scan_blind_mask & miss_mask).sum()),
    }


def generate_station_spurious(
    station_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    station_name: str,
    station_cfg: Dict[str, Any],
    scenario_id: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """按站级杂散比例生成假脉冲。"""
    spurious_count = int(np.rint(len(station_df) * float(station_cfg["spurious_ratio"])))
    if spurious_count <= 0:
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)

    if not station_df.empty:
        toa_low, toa_high = float(station_df["TOA"].min()), float(station_df["TOA"].max())
        rf_low, rf_high = float(station_df["RF"].min()), float(station_df["RF"].max())
        pw_low, pw_high = float(station_df["PW"].min()), float(station_df["PW"].max())
        pa_low, pa_high = float(station_df["PA"].min()), float(station_df["PA"].max())
    else:
        toa_low = 0.0
        toa_high = max(1.0, float(truth_df["toa_true_us"].max()) if not truth_df.empty else 1.0)
        rf_low, rf_high = 2_000.0, 12_000.0
        pw_low, pw_high = 0.2, 40.0
        pa_low, pa_high = -40.0, -10.0

    spurious = pd.DataFrame(
        {
            "TOA": rng.uniform(toa_low, toa_high, size=spurious_count),
            "RF": rng.uniform(rf_low, rf_high, size=spurious_count),
            "PW": rng.uniform(max(0.05, pw_low), max(pw_low + 0.05, pw_high), size=spurious_count),
            "PA": rng.uniform(pa_low, pa_high, size=spurious_count),
            "DOA": rng.uniform(0.0, 360.0, size=spurious_count),
            "Label": np.full(spurious_count, -1, dtype=int),
            "SNR": float(station_cfg["snr_db_mean"]) + rng.normal(-4.0, 2.5, size=spurious_count),
            "station_id": station_name,
            "scenario_id": scenario_id,
            "receiver_mode": station_cfg["receiver_mode"],
            "emitter_id": np.full(spurious_count, -1, dtype=int),
            "resolution_bin_toa": np.full(spurious_count, -1, dtype=int),
            "resolution_bin_rf": np.full(spurious_count, -1, dtype=int),
            "resolution_bin_pw": np.full(spurious_count, -1, dtype=int),
            "resolution_bin_pa": np.full(spurious_count, -1, dtype=int),
        }
    )
    spurious.sort_values("TOA", inplace=True)
    spurious.reset_index(drop=True, inplace=True)
    return spurious[OBSERVATION_COLUMNS]


def write_station_csvs(station_outputs: Dict[str, pd.DataFrame], output_dir: Path) -> Dict[str, str]:
    """将三站观测写入 generated_data。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files: Dict[str, str] = {}
    for station_name, frame in station_outputs.items():
        save_path = output_dir / f"{station_name}_data.csv"
        frame.to_csv(save_path, index=False)
        output_files[f"{station_name}_csv"] = str(save_path)
    return output_files


def write_summary(summary: Dict[str, Any], output_dir: Path) -> Path:
    """写出最小摘要文件。"""
    summary_path = output_dir / "external_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(to_serializable(summary), file_obj, indent=2, ensure_ascii=False)
    return summary_path


def run(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    # 1. 读取 12 个纯净训练文件，构建原始单类辐射源数据表。
    # 2. 根据配置选中若干辐射源，并为每个辐射源随机截取指定时间长度。
    # 3. 合并真值流后按时间切分 train/val/test，再选择一个切分作为当前输出场景。
    # 4. 将真值场景投影到 S1/S2/S3 三站，加入几何传播、时统误差、漏检和杂散。
    # 5. 写出三站 CSV 和最小摘要文件到 datasets/generated_data。
    scenario_cfg = dict(CONVERTER_CONFIG if config is None else config)
    seed = int(scenario_cfg["seed"])
    rng = np.random.default_rng(seed)

    source_map = load_clean_emitter_files(Path(scenario_cfg["source_train_dir"]))
    selected_emitters = select_emitters(scenario_cfg)

    truth_frames: List[pd.DataFrame] = []
    emitter_summary: List[Dict[str, Any]] = []
    for emitter_id, emitter_cfg in enumerate(selected_emitters):
        source_label = int(emitter_cfg["source_label"])
        truth_df, emitter_meta = build_emitter_truth_slice(
            emitter_cfg,
            source_map[source_label],
            rng,
            emitter_id,
        )
        truth_frames.append(truth_df)
        emitter_summary.append(emitter_meta)

    merged_truth = merge_truth_emitters(truth_frames)
    split_frames, split_meta = split_truth_stream(merged_truth, scenario_cfg["split_ratios"])
    selected_split = str(scenario_cfg["selected_output_split"])
    if selected_split not in split_frames:
        raise ValueError(f"不支持的 selected_output_split: {selected_split}")

    scene_truth = split_frames[selected_split]
    if scene_truth.empty:
        raise ValueError(f"{selected_split} 切分为空，无法生成观测。")

    station_outputs: Dict[str, pd.DataFrame] = {}
    station_summary: Dict[str, Dict[str, Any]] = {}
    for station_name, station_cfg in scenario_cfg["stations"].items():
        station_rng = np.random.default_rng(seed + sum(ord(char) for char in station_name) * 17)
        detected_truth_df, detect_stats = project_truth_to_station(
            scene_truth,
            station_name,
            station_cfg,
            scenario_cfg["scenario_id"],
            station_rng,
        )
        spurious_df = generate_station_spurious(
            detected_truth_df,
            scene_truth,
            station_name,
            station_cfg,
            scenario_cfg["scenario_id"],
            station_rng,
        )
        combined = pd.concat([detected_truth_df, spurious_df], ignore_index=True)
        combined.sort_values("TOA", inplace=True)
        combined.reset_index(drop=True, inplace=True)
        station_outputs[station_name] = combined[OBSERVATION_COLUMNS]
        station_summary[station_name] = {
            "coord": [float(v) for v in station_cfg["coord"]],
            "receiver_mode": station_cfg["receiver_mode"],
            "snr_db_mean": float(station_cfg["snr_db_mean"]),
            "miss_ratio": float(station_cfg["miss_ratio"]),
            "spurious_ratio": float(station_cfg["spurious_ratio"]),
            "time_sync_error_ns": float(station_cfg["time_sync_error_ns"]),
            "detected_truth_pulses": int(detect_stats["detected_truth_pulses"]),
            "spurious_pulses": int(len(spurious_df)),
            "output_rows": int(len(combined)),
            "scan_dropped_pulses": int(detect_stats["scan_dropped_pulses"]),
            "random_missed_pulses": int(detect_stats["random_missed_pulses"]),
        }

    output_dir = Path(scenario_cfg["output_dir"])
    output_files = write_station_csvs(station_outputs, output_dir)
    summary = {
        "scenario_id": scenario_cfg["scenario_id"],
        "source_dataset": "framist/2nd-EBDSC",
        "source_train_dir": scenario_cfg["source_train_dir"],
        "selected_emitter_files": [item["source_file"] for item in emitter_summary],
        "selected_emitter_labels": [item["source_label"] for item in emitter_summary],
        "selected_output_split": selected_split,
        "split_ratios": scenario_cfg["split_ratios"],
        "split_meta": split_meta,
        "total_truth_pulses": int(len(scene_truth)),
        "emitters": emitter_summary,
        "stations": station_summary,
        "output_files": output_files,
    }
    summary_path = write_summary(summary, output_dir)
    output_files["external_dataset_summary_json"] = str(summary_path)
    summary["output_files"] = output_files
    write_summary(summary, output_dir)
    logging.info(
        "External dataset conversion completed: %s, truth pulses=%d.",
        scenario_cfg["scenario_id"],
        len(scene_truth),
    )
    return summary


def main() -> Dict[str, Any]:
    """主入口：执行 2nd-EBDSC 纯净辐射源到三站观测 CSV 的转换。"""
    return run()


if __name__ == "__main__":
    main()



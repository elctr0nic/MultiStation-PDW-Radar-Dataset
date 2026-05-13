# MultiStation-PDW 雷达数据集

简介
- 本项目用于生成与可视化多站（multi-station）雷达脉冲描述词（PDW）模拟数据集，方便研究、算法开发与复现。

主要功能
- 多站雷达信号的 PDW 数据生成
- 可配置的仿真参数（参见 `pdw_sim_config.py`）
- 数据生成流水线 `pdw_sim_pipeline.py` 与可视化示例 `mulstation_pdw__visualize.py`

仓库结构（示例）
- `pdw_sim_config.py`：仿真参数配置
- `pdw_sim_pipeline.py`：数据生成流水线入口
- `existing_data_radar_loader.py`：数据读取器/示例
- `mulstation_pdw__visualize.py`：可视化脚本
- `output_visualization/`：示例输出目录

快速开始
1. 建议在虚拟环境中安装依赖：

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install numpy scipy matplotlib pandas
```

2. 生成数据集（默认配置）：

```bash
python pdw_sim_pipeline.py
```

3. 运行可视化示例：

```bash
python mulstation_pdw__visualize.py
```

数据格式说明（简要）
- 输出通常为 CSV 或可配置的二进制/文本格式，每条 PDW 记录包含：时间戳（timestamp）、幅度（amplitude）、频率（frequency）、发射站 ID（station_id）等字段。
- 详细字段请参考 `existing_data_radar_loader.py`。

自定义仿真
- 编辑 `pdw_sim_config.py` 来修改：站点数量、信噪比（SNR）、目标数、杂波/干扰模型、采样时长等。

许可证
- 本项目使用 MIT 许可证，见 `LICENSE`，请将 `LICENSE` 中的版权所有者替换为你的名字。

贡献指南
- 欢迎提交 issue 或 PR。请在 PR 中包含复现步骤与最小示例代码。


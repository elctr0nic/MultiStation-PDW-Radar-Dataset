# MultiStation-PDW Radar Dataset

Overview
- A toolkit to generate and visualize multi-station PDW (pulse descriptor word) radar simulation datasets for research and algorithm development.

Key features
- Multi-station PDW data generation
- Configurable simulation parameters (`pdw_sim_config.py`)
- Data generation pipeline (`pdw_sim_pipeline.py`) and visualization example (`mulstation_pdw__visualize.py`)

Repository layout (example)
- `pdw_sim_config.py`: simulation configuration
- `pdw_sim_pipeline.py`: pipeline entry for dataset generation
- `existing_data_radar_loader.py`: data loader example
- `mulstation_pdw__visualize.py`: visualization script
- `output_visualization/`: example outputs

Quick start
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install numpy scipy matplotlib pandas
```

2. Generate the dataset (default config):

```bash
python pdw_sim_pipeline.py
```

3. Visualize an example output:

```bash
python mulstation_pdw__visualize.py
```

Data format (brief)
- Outputs are typically CSV or configurable binary/text formats. Each PDW record contains timestamp, amplitude, frequency, station_id, etc.
- Refer to `existing_data_radar_loader.py` for exact field names and loading examples.

Customization
- Edit `pdw_sim_config.py` to change number of stations, SNR, targets, clutter/interference models, duration, and other parameters.

License
- This project uses the MIT License (see `LICENSE`). Replace the holder in `LICENSE` with your name.

Contributing
- Issues and PRs are welcome. Include reproducible steps and minimal examples in PRs.


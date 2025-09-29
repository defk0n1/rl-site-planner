# LTE Site Planning with Reinforcement Learning

An intelligent system for optimizing LTE cellular tower placement using Graph Neural Networks (GNN) and Proximal Policy Optimization (PPO). This project uses deep reinforcement learning to automatically determine optimal transmitter locations, configurations, and parameters for maximum coverage while minimizing interference.

## 🚀 Features

- **Graph Neural Network Policy**: Uses GNN architecture to model spatial relationships between candidate sites
- **PPO Training**: Stable reinforcement learning algorithm for policy optimization
- **Real-world Clutter Data**: Supports various terrain types (urban, suburban, rural) with realistic propagation models
- **Batch Processing**: Process multiple project areas simultaneously
- **Coverage Analysis**: RSRP (Reference Signal Received Power) and SINR (Signal-to-Interference-plus-Noise Ratio) calculations
- **Geographic Export**: Export results to CSV with real-world coordinates
- **Grid Generation**: Create tiling grids for large-scale deployment planning

## 📋 Requirements

```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib
pip install geopandas shapely pyproj rasterio
pip install simplekml requests affine
pip install tqdm
```

## 📊 Data Preparation

### Processing KMZ/KML Project Files

If you have geographic boundary files (KMZ/KML), convert them to project data:

```bash
python load_batch_polygon.py
```

**Requirements:**
- `index_clutter.txt`: Clutter tile index file
- `sq_centroid_kmz/`: Directory containing KMZ/KML boundary files
- Clutter data tiles referenced in the index

**Output:**
```
polygon_data/
├── project_1/
│   ├── mosaic.npy      # Processed clutter map
│   └── metadata.json   # Coordinate transform data
├── project_2/
│   ├── mosaic.npy
│   └── metadata.json
└── ...
```

### Generating Grid Tiles

Create systematic grid coverage for large areas:

```bash
python tile_gen.py --tile-area-km2 100 --grid square --mode clip
```

**Options:**
- `--tile-area-km2`: Target area per tile in km²
- `--grid`: Grid type (`square` or `hex`)
- `--mode`: Tile selection (`clip`, `within`, `centroid`)
- `--outdir`: Output directory for KML/KMZ files
- `--boundary`: Custom boundary file (optional)

**Example for Libya:**
```bash
python tile_gen.py --tile-area-km2 50 --grid hex --mode centroid --outdir libya_tiles
```

### Clutter Classes

The system supports 21 different clutter types:

```python
clutter_lookup = {
    0: ("urban", 0),      # Urban, no additional loss
    1: ("rural", 1),      # Rural, 1 dB loss
    2: ("rural", 0),      # Rural, no additional loss
    # ... (see code for full mapping)
    20: ("urban", 1.5),   # Urban, 1.5 dB loss
    -1: ("outofbounds", -1)  # Out of bounds
}
```

## 🏗️ Project Structure

```
├── run_simulation.py       # Main training script
├── testing.py             # Single project testing
├── batch_testing.py       # Batch processing for multiple projects
├── load_batch_polygon.py   # Process KMZ/KML files into project data
├── tile_gen.py            # Generate geographic grid tiles
├── load_clutter_data.py    # Load and process clutter maps
├── lte_env.py             # LTE environment (not shown but referenced)
├── gnn_models.py          # GNN policy and value networks (not shown)
├── lte_utils.py           # Utility functions for RSRP/SINR (not shown)
├── visualise.py           # Visualization utilities (not shown)
└── metrics_tracker.py     # Training metrics tracking (not shown)
```

## 🎯 Quick Start

### 1. Training a New Model

Train the RL agent from scratch:

```bash
python run_simulation.py
```

This will:
- Load clutter data from `rastermap.npy`
- Initialize GNN policy and value networks
- Train using PPO for 1000 episodes
- Save model checkpoints every 10 episodes
- Generate training plots and metrics

**Key hyperparameters:**
- Learning rate: 3e-4
- Clip epsilon: 0.5
- Gamma (discount): 0.99
- Batch size: 64

### 2. Testing a Trained Model

Test a trained model on a specific project:

```bash
python testing.py --path /path/to/trained/model.pth
```

**Example:**
```bash
python testing.py --path experiments/run_20240308_143022/ppo_gnn_policy_100.pth
```

This will:
- Load the trained model
- Process the project data in `polygon_data/`
- Generate transmitter placement recommendations
- Export results to CSV with real-world coordinates
- Display coverage maps and statistics

### 3. Batch Processing Multiple Projects

Process multiple project areas simultaneously:

```bash
python batch_testing.py --model /path/to/model.pth --base_dir polygon_data
```

**Example:**
```bash
python batch_testing.py --model experiments/run_20240308_143022/ppo_gnn_policy_100.pth --base_dir polygon_data
```

This will:
- Process all projects in the `polygon_data/` directory
- Generate individual results for each project
- Create aggregated summary statistics
- Save failed projects log for debugging

## 🎛️ Configuration

### Action Space

The RL agent controls 4 parameters for each candidate site:

1. **Placement** (Binary): Whether to place a transmitter
2. **Height** (Continuous): Antenna height (28-45 meters)
3. **Tilt** (Continuous): Electrical downtilt (0-12 degrees)
4. **Azimuth** (Discrete): Antenna azimuth (0-360°, 5° steps)

### Environment Parameters

Default parameters (can be modified in the LTE environment source code):
- **Frequency**: 1800 MHz (LTE Band 3)
- **TX Power**: 46 dBm (40W)
- **Resolution**: 50m pixel resolution
- **Candidate spacing**: 10m grid (training), 25m grid (testing)

### Modifying Network Architecture

Edit the GNN models in `gnn_models.py` to change:
- Number of GNN layers
- Hidden dimensions
- Message passing functions
- Output heads

### Adjusting Training Parameters

In `run_simulation.py`, modify:
- `LEARNING_RATE`: Learning rate for optimizer
- `CLIP_EPSILON`: PPO clipping parameter
- `GAMMA`: Discount factor
- `BATCH_SIZE`: Training batch size
- `NUM_EPISODES`: Total training episodes

### Custom Clutter Data

Replace `rastermap.npy` with your own clutter data:
1. Ensure data is in `uint16` format
2. Values should match the clutter lookup table
3. Update coordinate transform information

## 📈 Output Files

### Training Outputs

```
ppo_logs/run_YYYYMMDD_HHMMSS/
├── ppo_gnn_policy_X.pth    # Policy network weights
├── ppo_gnn_value_X.pth     # Value network weights
├── metrics_ep_X.png        # Training metrics plot
└── training_logs.csv       # Detailed training data
```

### Testing Outputs

**Single project:**
```
polygon_data/project_name/
├── mosaic.npy
├── metadata.json
└── deployed_transmitters_YYYYMMDD_HHMMSS.csv
```

**Batch processing:**
```
polygon_data/aggregated_results/
├── all_transmitters_YYYYMMDD_HHMMSS.csv  # All transmitters
├── summary_YYYYMMDD_HHMMSS.csv           # Project summaries
└── failed_projects.csv                    # Failed projects log
```

### CSV Output Format

```csv
Transceiver Name,Cell Name,Longitude,Latitude,Frequency Band,Scenario,Antenna Type,Azimuth,Height,Mechanical Downtilt,Electrical Downtilt,Max Power(dBm)
PLACEHOLDER,PLACEHOLDER_0,15.123456,32.654321,1800,urban,1800MHz 65deg 17dBi 0Tilt,45.0,35.2,0,8.5,46
```

## 📊 Performance Metrics

The system tracks several key metrics:

- **Coverage**: Percentage of area with RSRP > -95dBm and > -100dBm
- **Efficiency**: Number of transmitters deployed
- **Reward**: Composite score balancing coverage and cost
- **Training metrics**: Policy loss, value loss, entropy

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **No valid candidates**: Check clutter map boundaries
3. **Import errors**: Verify all dependencies are installed
4. **Coordinate mismatch**: Ensure consistent CRS (EPSG:32633)

### Debug Mode

Enable verbose logging by adding debug prints in the training loop:

```python
print(f"State shape: {state.shape}")
print(f"Action: {action}")
print(f"Reward: {reward}")
```

## 📚 References

- Proximal Policy Optimization (PPO): [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
- Graph Neural Networks: [arXiv:1901.00596](https://arxiv.org/abs/1901.00596)
- LTE Radio Planning: ITU-R Recommendations

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

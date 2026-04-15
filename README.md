# CARLA Garage Ablation Studies

This repository contains cleaner versions of the code used during the creation of this project, including training scripts, activation extraction tools, and data collection utilities for CARLA autonomous driving experiments.

## Project Structure

### New Files Added

| File | Purpose |
|------|---------|
| `TFpp_README.md` | Documentation for TransFuser++ (renamed from original README.md) |
| `collect_dataset_local.py` | Script for local dataset collection from CARLA routes |
| `team_code/activation_extractor.py` | Extracts and visualizes model layer activations |
| `team_code/bash_extract_activation.sh` | Shell script to run activation extraction |
| `team_code/chunked_shell_train.sh` | Robust chunked training with retry logic |
| `team_code/config.py` | Global configuration class with all hyperparameters |

---

## File Descriptions

### collect_dataset_local.py

Script for collecting training data locally from CARLA routes.

**Purpose:** Randomly samples routes from the dataset folder and runs the CARLA leaderboard evaluator in data generation mode to collect sensor data.

**Configuration:**
- `code_root`: Path to carla_garage
- `carla_root`: Path to CARLA installation
- `data_save_directory`: Where to save collected data
- `route_folder`: Folder containing route XML files

**Key Parameters:**
- `N_SCENARIOS = 1000`: Number of scenarios to collect
- `COLLECTED_SCENARIOS = 0`: Offset for partial collection
- Fixed ports for parallel CARLA instances: 10010, 20010, 30010

**Usage:**
```bash
python collect_dataset_local.py
```

---

### team_code/activation_extractor.py

Extracts and visualizes intermediate layer activations from TransFuser++ models.

**Classes:**

1. **ActivationExtractor** - Registers forward hooks on model layers to capture activations
   - Hooks into: image encoder, transformer fusion layers, lidar encoder, fusion layers, global pooling
   - Methods: `register_hooks()`, `clear_activations()`, `remove_hooks()`

2. **ActivationVisualizer** - Saves activations as images and numpy arrays
   - Saves input RGB images alongside activation heatmaps
   - Supports overlay-only mode (faster, smaller output)
   - Saves config metadata for reference

**Command Line Arguments:**
```
--id                  Unique experiment identifier
--logdir               Directory to log data and models to
--load_file            Model to load for initialization
--root_dir             Root directory of training data
--cpu_cores            Number of CPU cores available
--sample_indices       Comma-separated sample indices to process
--save_activations     Directory to save activation images
--save_activations_np  Directory to save activations as numpy
--overlay_only        Only save overlay heatmaps (skip raw channels)
--batch_size          Batch size for extraction
--num_batches         Number of batches to process
--seed                Random seed
--config_json          Path to config.json file
```

**Usage:**
```bash
python team_code/activation_extractor.py \
    --logdir /path/to/output \
    --load_file /path/to/model.pth \
    --root_dir /path/to/dataset \
    --cpu_cores 12 \
    --batch_size 20 \
    --num_batches 20 \
    --overlay_only 1 \
    --save_activations /path/to/activations \
    --sample_indices 0,2,4,6,8,10 \
    --config_json /path/to/config.json \
    --seed 0
```

---

### team_code/bash_extract_activation.sh

Shell script to run activation extraction with predefined settings.

**Purpose:** Simplified interface for extracting activations from trained models.

**Configuration:**
- CARLA paths
- Model weights location
- Dataset root
- Sample indices for extraction (12 samples by default)
- Config JSON for model architecture

**Default Settings:**
- Batch size: 20
- Num batches: 20
- Overlay only: enabled

---

### team_code/chunked_shell_train.sh

Robust training script with chunked epochs and automatic retry on failure.

**Purpose:** Handles long training runs with checkpoint detection and resume capability.

**Features:**
- Chunked training (trains in chunks of N epochs, configurable)
- Automatic retry on failure (max 5 retries)
- Checkpoint detection and loading
- Support for continuing from previous runs

**Key Configuration:**
```bash
ID="tfpp_0010_0"
LOGDIR="/media/.../augmented_model"
ROOT_DIR="/path/to/dataset"
TARGET_EPOCHS=31
CHUNK=2
MAX_RETRIES=5
```

**Training Arguments:**
- `--backbone transFuser`
- `--image_architecture regnety_008`
- `--lidar_architecture regnety_008`
- `--n_layer 1`
- `--use_velocity 1`
- `--use_controller_input_prediction 1`
- `--lr 0.0002`
- `--batch_size 7`
- `--setting all`
- Many more in BASE_ARGS array

**Usage:**
```bash
bash team_code/chunked_shell_train.sh
```

**Or directly with Python:**
```bash
python -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 \
    team_code/train.py \
    --id tfpp_0010_0 \
    --lr 0.0002 \
    --batch_size 7 \
    --root_dir /path/to/dataset \
    --logdir /path/to/logs \
    --epochs 31
```

---

### team_code/config.py

Comprehensive configuration class (`GlobalConfig`) containing all hyperparameters.

**Sections:**

1. **Autopilot Parameters**
   - Frame rate, noise, detection radius
   - IDM parameters for vehicle/pedestrian/traffic light control
   - Bicycle model parameters

2. **Controller Parameters**
   - Longitudinal Linear Regression controller
   - Longitudinal PID controller
   - Lateral PID controller
   - Kinematic Bicycle Model

3. **Sensor Configuration**
   - LiDAR: position, rotation, points_per_second
   - Camera: position, rotation, resolution, FOV

4. **DataLoader Settings**
   - Sequence lengths, resolution, crop settings
   - LiDAR range, pixel density
   - Semantic/depth classes

5. **Training Hyperparameters**
   - Learning rate, batch size, epochs
   - Augmentation options
   - Loss weights
   - Optimizer settings

6. **TransFuser Model Architecture**
   - Fusion mode (both/image_only/lidar_only)
   - Attention gating
   - Channel swapping
   - Transformer layers, heads
   - Auxiliary tasks (semantic, depth, detection)

---

## Usage Examples

Before training your models. make sure to `TFpp.md` for detailed setup and execution. It is needed to customize some paths to where you installed your CARLA instance and the python environment used. As well as installing required packages.

### Training a Model
```bash
cd team_code
bash chunked_shell_train.sh
```

### Extracting Activations for Analysis
```bash
cd team_code
bash bash_extract_activation.sh
```

### Collecting Data Locally
```bash
python collect_dataset_local.py
```

---

## Dependencies

- CARLA 0.9.15
- Python 3.10+
- PyTorch
- numpy, opencv-python
- tqdm
- jsonpickle

---

## Citation

If you use this code, please cite the original CARLA Garage papers:

```bibtex
@InProceedings{Jaeger2023ICCV,
  title={Hidden Biases of End-to-End Driving Models},
  author={Bernhard Jaeger and Kashyap Chitta and Andreas Geiger},
  booktitle={ICCV},
  year={2023}
}
```
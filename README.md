# TrajectoryInit

TrajectoryInit is a Python package for initializing trajectory priors from MOT tracks and generating scene-level outputs such as averaged trajectories and zone metadata.

<!-- It currently includes:
- Track filtering and preprocessing
- Occlusion-zone inference
- Trajectory initialization pipeline via `Trajectory_Initialisation`
- Feature-extractor components and model config assets -->

## Installation

### Poetry (recommended)

Choose compatible version for your system from https://download.pytorch.org and replace the source link
```bash
poetry source add --priority=explicit pytorch-gpu https://download.pytorch.org/whl/cu128

poetry lock
```

```bash
poetry install
```

### Fallback venv

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Running the Pipeline

The helper scripts do the following:
- Create/use Poetry environment when available
- Fall back to local `.venv` when Poetry is unavailable
- Create `Trajectory_Initialisation`, call `.run()`, then call `.save_out(...)`
- Write logs to `output_logs/`

### Linux/macOS / Git Bash

```bash
chmod +x ./run_pipeline.sh
./run_pipeline.sh ./data/labels.txt ./output_json
```

### Windows PowerShell

```powershell
.\run_pipeline.ps1 -LabelPath ".\data\labels.txt" -OutputDir ".\output_json"
```


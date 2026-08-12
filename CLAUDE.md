# CLAUDE.md

## Project Overview

**fujin_uav_slam** is a visual SLAM (Simultaneous Localization and Mapping) system for UAV (drone) applications. It processes monocular camera feeds from ROS bag files to perform feature-based visual odometry. The project is early-stage — the frontend (feature extraction, matching, visualization) is functional, while backend modules (bundle adjustment, loop closure, mapping) are scaffolded but not yet implemented.

## Tech Stack

- **Language:** Python 3.10
- **Core libraries:** OpenCV 4.10, NumPy, PyYAML
- **Data formats:** ROS bag files, EuRoC MAV dataset (ETH ASL)
- **Testing:** pytest (no tests written yet)
- **Dependencies:** managed via `requirements.txt` with pinned versions

## Project Structure

```
src/
├── main.py                      # Entry point — orchestrates pipeline, CLI args, visualization
├── frontend/                    # Visual odometry (implemented)
│   ├── visual_odometry.py       # Main VO pipeline orchestrator
│   ├── feature_extraction.py    # ORB and SIFT extractors (Strategy pattern)
│   └── feature_matching.py      # BFMatcher and FLANNMatcher (Strategy pattern)
├── input_process/               # Dataset loading
│   ├── process_EuRoC_bag.py     # ROS bag streaming via multiprocessing
│   └── process_EuRoC_asl.py     # EuRoC ASL CSV-based loader
├── backend/                     # Optimization (not yet implemented)
├── mapping/                     # 3D map construction (not yet implemented)
├── tracking/                    # Long-term tracking (not yet implemented)
└── utils/                       # Utilities (not yet implemented)
config/
└── config.yaml                  # Debug flag config
data/                            # Dataset files (gitignored)
test/                            # Test directory (empty)
notebooks/                       # Jupyter notebooks for exploration
```

## Running the Project

```bash
# Activate virtual environment
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the SLAM pipeline
python3 src/main.py -bag path/to/file.bag -topic /cam0/image_raw --debug
```

**CLI arguments:**
- `-bag` (required): path to a `.bag` file
- `-topic` (optional, default: `/cam0/image_raw`): ROS bag topic with mono video frames
- `--debug` (optional): enables feature match visualization overlay

The pipeline opens an OpenCV window showing the processed frames. Press `q` to quit.

## Architecture Notes

- **Strategy pattern** is used for swappable feature extractors (`ORBFeatureExtractor`, `SIFTFeatureExtractor`) and matchers (`BFMatcher`, `FLANNMatcher`), all deriving from abstract base classes.
- **Multiprocessing** is used for bag file streaming (`BagImageStreamer`) to avoid blocking the main processing thread. Frames are passed through a `Queue`.
- The pipeline processes consecutive frame pairs: extract features -> match features -> (motion estimation is stubbed out).
- `VisualOdometry` has helper methods `add_ones()` for homogeneous coordinates and `triangulate()` using SVD.
- `main.py` contains significant commented-out reference code (optical flow experiments, earlier pipeline versions).

## Development Guidelines

- Run from the repo root (`src/main.py` expects `config/config.yaml` via relative path).
- The `data/` directory is gitignored — dataset files are not committed.
- The `env/` virtual environment directory is also gitignored.
- No CI/CD is configured. No tests exist yet — `test/` is empty.
- When adding new feature extractors or matchers, follow the existing abstract base class pattern in `feature_extraction.py` and `feature_matching.py`.

## Testing

```bash
pytest test/
```

No tests currently exist. When adding tests, use pytest and place them in `test/`.

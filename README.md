# fujin_uav_slam

<!-- ## Simulator Pre-Requisits
- Install the jMAVSim Simulator (macOS)
    - Install a recent version of Java usinf Eclipse Temurin
    `brew install px4-sim-jmavsim`
    - Install jMAVSim
    `brew install px4-sim-jmavsim` -->

## SLAM

### Frontend module
Estimate campera pose and map the points in 3D
**Pipeline**:
- Feature extraction
- Feature matching
- Pose estimation
- Triangulation
- Local mapping 

### Backend module
Optimize the estimated camera pose and mapping
**Pipeline**:
-  Bundle adjustment
- Loop closure detection
- Pose graph optimization

### Run SLAM pipeline
`python3 src/main.py -bag path/to/file.bag -topic mono/stream/topic --debug`

**Args**:
`-bag` - path to .bag file recording
`-topic` - bag file topic containing mono video frames
`--debug` - flag for running the SLAM in debug mode, display useful pipeline process information
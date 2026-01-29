# Structure from Motion Implementation

3D reconstruction from 2D image sequences using incremental Structure from Motion pipeline.

## Overview

This project implements a complete SfM pipeline that reconstructs 3D scenes from multiple 2D images. Developed as part of the Computer Vision course at Chalmers University of Technology.

## Features

- SIFT feature extraction and matching
- Parallel RANSAC for Essential and Homography matrix estimation
- Handles both general and planar scenes
- Rotation chaining with SO(3) projection
- Translation estimation from 2D-3D correspondences
- Point cloud visualization with Open3D

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from sfm_pipeline import run_sfm
from project_helpers import get_dataset_info

# Run SfM on a dataset
K, img_names, init_pair, pixel_threshold = get_dataset_info(dataset=3)
point_cloud, cameras = run_sfm(img_names, K, init_pair, pixel_threshold)
```

## Results

Successfully tested on 7 datasets including:
- Cathedral gates
- Fountain scenes
- Golden statue
- Planar building facades
- Relief sculptures

See the full report in `Computer_Vision_Project_Ayberk_Tunca.pdf`

## Requirements

- Python 3.11+
- OpenCV
- NumPy
- SciPy
- Open3D

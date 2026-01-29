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

## Author

Ayberk Tunca - Chalmers University of Technology

## Course

Computer Vision Project (January 2026)
Instructor: Professor Fredrik Kahl
```

**requirements.txt**
```
jupyter>=1.0.0
numpy>=1.24.0
opencv-python>=4.8.0
scipy>=1.11.0
pillow>=10.0.0
matplotlib>=3.7.0
ipympl>=0.9.0
open3d>=0.17.0
```

**.gitignore**
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data files
data/
*.JPG
*.jpg
*.jpeg
*.png
*.gif
*.bmp
*.tif
*.tiff

# Output files
output/
recon_output/
*.ply
*.obj
*.pcd

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Cache
.cache/
cache/
__pycache__/
.venv/
.pytest_cache/

# OS
Thumbs.db
.DS_Store

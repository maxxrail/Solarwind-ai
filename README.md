# Solarwind-ai
Below is a **single Markdown document** containing everything you need. It’s entirely in Markdown format, including headings, lists, code fences, etc. Copy this into a file named **`README.md`** (or any Markdown file) for a complete overview.

---

# Rooftop Detection with Detectron2

This repository demonstrates how to use [Detectron2](https://github.com/facebookresearch/detectron2) to detect and segment rooftops (buildings) in aerial imagery, focusing on the [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/).

---

## Table of Contents

1. [Introduction](#introduction)  
2. [Requirements](#requirements)  
3. [Project Structure](#project-structure)  
4. [Dataset Preparation](#dataset-preparation)  
5. [Setting Up the Environment](#setting-up-the-environment)  
6. [Key Configuration Parameters](#key-configuration-parameters)  
7. [Training](#training)  
8. [Inference](#inference)  
9. [Tips & Troubleshooting](#tips--troubleshooting)  
10. [References](#references)

---

## Introduction

The **Inria Aerial Image Labeling Dataset** provides large aerial images, each with ground-truth building (rooftop) masks. We use **Mask R-CNN** (a model for instance segmentation) within **Detectron2** to learn how to segment these rooftops.

The structure of the dataset is as follows and you can **download the dataset from our google drive**
AerialImageDataset/
├── train/
│   ├── images/
│   │   ├── 0.tif
│   │   ├── 1.tif
│   │   └── ...
│   └── gt/
│       ├── 0.tif
│       ├── 1.tif
│       └── ...
└── val/
│   ├── images/
│   │   └── ...
│   └── gt/

**Goal**  
- Detect & segment building footprints (rooftops) from aerial images.  
- Use a single class, `"roof"`, to keep it simple.

---

## Requirements

1. **Python 3.8+**  
   - Python 3.11 may not be fully supported by all libraries, so 3.9 or 3.10 is safer.
2. **PyTorch**  
   - CPU or GPU version (NVIDIA GPU recommended for faster training).
3. **Detectron2**  
   - Official binaries are for NVIDIA GPUs (CUDA). For CPU-only, see below.
4. **Inria Aerial Dataset**  
   - Download and extract from [Inria website](https://project.inria.fr/aerialimagelabeling/).
5. **Other Python packages**  
   - `opencv-python`, `matplotlib`, `numpy`, `tqdm`, etc.

**Note**  
If you have an **AMD Radeon GPU on Windows**, Detectron2 does not officially support ROCm on Windows. Either run on CPU or switch to an NVIDIA GPU (or advanced Linux+ROCm setup).

---

## Project Structure

SOLARWIND-AI/
├── detectron2/ (optional local clone, or installed via pip)
├── model.ipynb
└── README.md
```

You can adapt as needed, but the **key** is:

- **`images/`**: raw aerial images.  
- **`gt/`**: ground-truth masks with matching filenames.

---

## Setting Up the Environment

1. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv myenv
   source myenv/bin/activate  # or myenv\Scripts\activate (Windows)
   ```

2. **Install Libraries** 
   - All necessary libraries are located in the first cell of the **model.ipynb** file

 3. **Install Detectron2 (if not already installed)**  
   - if detectron2 isnt already located in the project please uncomment the cloning code and dont forget to MOVE THE CONTENTS FROM THE SUBFOLDER DETECTRON2 FOLDER TO THE MAIN DETECTRON2 FOLDER!
---

## Key Configuration Parameters

In model.ipynb:

```python
# Standard parameters
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 1000
```

### 1. `DATALOADER.NUM_WORKERS`
- **Definition**: The number of CPU processes used for loading data in parallel.  
- **Impact**: Higher means faster data loading but more CPU usage.

### 2. `SOLVER.IMS_PER_BATCH`
- **Definition**: Batch size (# images processed per iteration).  
- **Impact**: Larger batch can speed up GPU training (if memory allows). On CPU or limited VRAM, keep it low to avoid out-of-memory.

### 3. `SOLVER.BASE_LR`
- **Definition**: Base learning rate for gradient descent.  
- **Impact**: Too high can cause training instability; too low might slow convergence. 0.00025 is a decent starting point for small batch sizes.

### 4. `SOLVER.MAX_ITER`
- **Definition**: Total number of training iterations.  
- **Impact**: More iterations → more training time → potentially better accuracy. 1000 is usually a quick test.

---

## Tips & Troubleshooting

1. **CPU Training Is Slow**  
   - Consider cropping large 5000×5000 tiles into 512×512 or 1024×1024 patches.  
   - Keep `IMS_PER_BATCH` low if you’re running out of memory.

2. **ValueError: mask_format=='polygon'**  
   - Make sure your dataset returns valid polygons in double brackets, skipping degenerate contours.

3. **Incompatible Shapes (Skip loading parameter...)**  
   - Normal if you have 1 class vs. 80 classes from COCO. Detectron2 automatically re-initializes final layers.

4. **Running Out of Memory**  
   - Lower `IMS_PER_BATCH`.  
   - Crop images to smaller sizes.  
   - Decrease `NUM_WORKERS` if CPU usage is too high.

5. **AMD Radeon GPU on Windows**  
   - Not officially supported by Detectron2. Use CPU or an NVIDIA GPU. Linux + ROCm is advanced and not guaranteed.

---

## References

- [Detectron2 GitHub](https://github.com/facebookresearch/detectron2)  
- [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)  

---

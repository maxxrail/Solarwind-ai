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

The structure of the dataset is as follows:
AerialImageDataset/<br>
├── train/<br>
│   ├── images/<br>
│   │   ├── 0.tif<br>
│   │   ├── 1.tif<br>
│   │   └── ...<br>
│   └── gt/<br>
│       ├── 0.tif<br>
│       ├── 1.tif<br>
│       └── ...<br>
├── val/<br>
│   ├── images/<br>
│   │   └── ...<br>
│   └── gt/<br>
|   │   └── ...<br>
└── test/<br>
|   └── images/<br>
|   │   └── ...<br>

**Goal**  
- Detect & segment building footprints (rooftops) from aerial images.  

---

## Requirements

1. **Python 3.8-3.10**  
   - Python 3.11 may not be fully supported by all libraries, so 3.9 or 3.10 is safer.
2. **PyTorch**  
   - CPU or GPU version (NVIDIA GPU recommended for faster training).
   - Installation part of code.
3. **Detectron2**  
   - Official binaries are for NVIDIA GPUs (CUDA). For CPU-only, see below.
   - Installation part of code.
4. **Inria Aerial Dataset**  
   - Download and extract from [Google Drive](https://drive.google.com/drive/folders/1uM0dbL6uy0khDwD17VB70J8aOWIigQBm?usp=drive_link).
5. **Other Python packages**  
   - `opencv-python`, `matplotlib`, `numpy`, `tqdm`, etc.
6. **Visual C++ 14.0 or later**
   - Download from [Microsoft Website](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

**Note**  
If you have an **AMD Radeon GPU on Windows**, Detectron2 does not officially support ROCm on Windows. Either run on CPU or switch to an NVIDIA GPU (or advanced Linux+ROCm setup).

---

## Project Structure

SOLARWIND-AI/<br>
├── detectron2<br>
├── model.ipynb<br>
├── output **PLEASE DO NOT COMMIT THIS FOLDER**<br> 
└── README.md<br>
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

 3. **Install Detectron2 (if not already located in repo**  
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
2. **Incompatible Shapes (Skip loading parameter...)**  
   - Normal if you have 1 class vs. 80 classes from COCO. Detectron2 automatically re-initializes final layers.
3. **Running Out of Memory**  
   - Lower `IMS_PER_BATCH`.  
   - Crop images to smaller sizes.  
   - Decrease `NUM_WORKERS` if CPU usage is too high.

4. **AMD Radeon GPU on Windows**  
   - Not officially supported by Detectron2. Use CPU or an NVIDIA GPU. Linux + ROCm is advanced and not guaranteed.

---

## References

- [Detectron2 GitHub](https://github.com/facebookresearch/detectron2)  
- [Inria Aerial Image Labeling Dataset](https://project.inria.fr/aerialimagelabeling/)  

---

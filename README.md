

# HIPTrack (Historical Prompt Tracking)

HIPTrack is a state-of-the-art visual object tracking system that integrates historical prompt memory, text encoding for language grounding, and backward tracking mechanisms. This repository contains the official PyTorch implementation of the HIPTrack tracker.

## Introduction

HIPTrack is designed to handle complex tracking scenarios by maintaining a memory of historical features (prompts) and leveraging natural language descriptions of the target. It includes a dedicated backward tracking module for error correction and robust mask refinement capabilities.

### Key Features

*   **Historical Prompt Network (HIP):** Efficiently stores and retrieves temporal context to handle occlusions and target deformation.
*   **Text-Driven Tracking:** Utilizes a text encoder (e.g., RoBERTa) to integrate natural language target descriptions into the tracking process.
*   **Backward Tracking:** Implements a reverse-time verification step to identify and correct tracking errors.
*   **Mask Refinement:** Integrates Alpha-Refine (AR) for high-quality segmentation mask generation.
*   **Comprehensive Benchmark Support:** Supports training and evaluation on major tracking datasets like LaSOT, GOT-10k, TrackingNet, and more.

## Installation

### Prerequisites

*   Python 3.8+
*   PyTorch 1.9+
*   CUDA 11.1+ (Recommended)
*   Conda

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://gitee.com/Ningyaoa/bvl.git
    cd bvl
    ```

2.  **Create the conda environment:**
    ```bash
    conda env create -f environment.yml
    conda activate bvl
    ```

3.  **Install external dependencies:**
    This project relies on the Alpha-Refine (AR) library and PreciseRoIPooling.
    ```bash
    # Install PreciseRoIPooling
    cd external/PreciseRoIPooling/pytorch
    python setup.py develop
    cd ../../..

    # Install pytracking (part of AR)
    cd external/AR/pytracking
    # Note: You may need to create local.py files as described in external/AR/README.md
    python -c "from pytracking.libs import precision; print('pytracking libs OK')"
    cd ../..
    ```

## Model Zoo

Pre-trained models and configuration files are essential for running the tracker. You can find the experiment configurations in `experiments/hiptrack/`.

*   **Trackers Config:** `experiments/hiptrack/hiptrack.yaml`
*   **Training Configs:** `experiments/hiptrack/hiptrack_train_full.yaml`, etc.

## Usage

### Training

Training scripts are located in `lib/train/`. You can start training using the provided scripts.

```bash
# Example training command
python tracking/train.py --script hiptrack --config hiptrack_train_full
```

Ensure your `local.py` settings in `lib/train/admin/` and `external/AR/ltr/admin/` are correctly set up with paths to your datasets (e.g., LaSOT, GOT10k).

### Testing

To test the tracker on a video or a dataset, use the scripts in the `tracking/` directory.

**Test on a specific dataset (e.g., OTB):**

```bash
python tracking/test.py --tracker_name hiptrack --tracker_param hiptrack --dataset otb
```

**Test on a specific sequence:**

```bash
python tracking/test.py --tracker_name hiptrack --tracker_param hiptrack --dataset otb --sequence "Basketball"
```

**Video Demo:**

To run the tracker on a local video file:

```bash
python tracking/video_demo.py --tracker_name hiptrack --tracker_param hiptrack --videofile path/to/your/video.mp4
```

## Directory Structure

*   `lib/models/`: Core model definitions.
    *   `hiptrack/`: Main tracker architecture, Vision Transformer backbone, and box heads.
    *   `hip/`: Historical Prompt Network implementation.
    *   `layers/`: Custom neural network layers (LoRA, Attention, etc.).
    *   `text/`: Text encoder wrapper.
*   `lib/train/`: Training pipeline.
    *   `actors/`: Training loss definitions and forward passes.
    *   `dataset/`: Dataset loaders (LaSOT, COCO, etc.).
    *   `data/`: Data augmentation and processing.
*   `lib/test/`: Evaluation pipeline.
    *   `evaluation/`: Dataset interfaces for testing.
    *   `tracker/`: Tracker inference logic.
*   `external/`: Third-party libraries (Alpha-Refine, LTR, PreciseRoIPooling).
*   `tracking/`: High-level scripts for testing and demos.
*   `experiments/`: Configuration YAML files.

## Acknowledgments

This project builds upon several excellent open-source works. We thank the authors of:

*   **LTR (Learning Tracking Representations):** For the training framework and dataset loaders.
*   **Alpha-Refine (AR):** For the mask refinement module.
*   **PreciseRoIPooling:** For the precise ROI pooling operations.
*   **Vision Transformers:** For the backbone architecture.

## License

This project is licensed under the MIT License. See the source code for details.
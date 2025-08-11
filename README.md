# Deepfake Detection

A deep learning framework for audio deepfake detection using state-of-the-art models including AASIST and LCNN architectures.

## Features

- Support for multiple model architectures (AASIST, LCNN)
- Multi-GPU and single-GPU training support
- Comprehensive evaluation metrics
- Audio preprocessing and augmentation
- Pre-trained model checkpoints

## Environment Setup

### Prerequisites
- Python 3.9+
- CUDA-compatible GPU (recommended)
- Conda package manager

### Installation

1. Create and activate conda environment:
```bash
conda create -n deepfake python=3.6
conda activate deepfake
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Alternatively, you can use the provided conda environment file:
```bash
conda env create -f environment.yml
conda activate deepfake
```

## Dataset Preparation

### Data Format
Prepare your dataset with the following structure:

```
data/
├── train.txt
├── test.txt
└── audio_files/
    ├── file1.wav
    ├── file2.wav
    └── ...
```

### Label File Format
Create label files (train.txt, test.txt) with the following format:
```
file_id,file_path,label
0069156,aigc_speech_detection_tasks_part1/0069156.wav,Bonafide
0069157,aigc_speech_detection_tasks_part1/0069157.wav,Spoof
```

Where:
- `file_id`: Unique identifier for the audio file
- `file_path`: Relative path to the audio file
- `label`: Either "Bonafide" (real) or "Spoof" (fake)

## Training

### Multi-GPU Training
For training with multiple GPUs:
```bash
bash train.sh
```

### Single-GPU Training
For training with a single GPU:
```bash
bash run_single.sh
```

### Custom Training
You can also run training directly with custom parameters:
```bash
python train.py --config_path configs/your_config.yaml
```

## Inference

### Batch Inference
Run inference on a dataset:
```bash
python inference.py --model_path path/to/model.pth --data_path path/to/test_data
```

### Single File Inference
Use the provided shell script for quick inference:
```bash
bash infer.sh path/to/audio_file.wav
```

## Model Architecture

This project supports multiple architectures:
- **AASIST**: Audio Anti-Spoofing using Integrated Spectro-Temporal graph attention
- **LCNN**: Light Convolutional Neural Network

## Project Structure

```
├── module/                 # Model architectures
│   ├── AASIST.py          # AASIST model implementation
│   └── LCNN.py            # LCNN model implementation
├── scripts/               # Utility scripts
├── ckpts/                 # Pre-trained checkpoints
├── exports/               # Training outputs
├── dataset.py             # Dataset loading utilities
├── model.py               # Model wrapper
├── learner.py             # Training logic
├── inference.py           # Inference utilities
├── metrics.py             # Evaluation metrics
└── params.py              # Configuration parameters
```

## Results

Training logs and model checkpoints are saved in the `exports/` directory. You can monitor training progress and evaluate model performance using the provided metrics.

## Contributing

Feel free to submit issues and pull requests to improve this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.


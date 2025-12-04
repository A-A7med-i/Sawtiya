# Sawtiya - Audio Emotion Recognition

> A deep learning framework for recognizing emotions from audio using CNNs, BiLSTMs, and attention mechanisms.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Details](#model-details)
- [Contributing](#contributing)
- [License](#license)

## Overview

Sawtiya is a comprehensive audio processing and emotion recognition toolkit that combines state-of-the-art deep learning techniques with robust audio feature extraction. The system processes raw audio files, extracts meaningful features, and classifies emotions using a hybrid CNN-BiLSTM architecture with attention mechanisms.

### Key Capabilities

- **Metadata Extraction**: Automated parsing of structured audio filenames
- **Audio Preprocessing**: Silence trimming, normalization, and resampling
- **Feature Engineering**: Mel-spectrograms, MFCCs, and spectral features
- **Data Augmentation**: Time-stretching, pitch-shifting, and noise injection
- **Deep Learning Pipeline**: End-to-end training with PyTorch
- **Visualization Tools**: Comprehensive audio analysis and feature visualization

## Features

- **Advanced Audio Processing**: Handles variable-length audio with intelligent padding
- **Hybrid Architecture**: Combines CNN spatial features with BiLSTM temporal modeling
- **Attention Mechanism**: Focuses on emotionally significant audio segments
- **Rich Visualizations**: Waveforms, spectrograms, Mel-spectrograms, and MFCCs
- **Data Augmentation**: Robust training through audio transformations
- **GPU Acceleration**: Full CUDA support for fast training
- **Training Monitoring**: Real-time metrics and checkpoint management

## Architecture

```
Audio Input → Preprocessing → Feature Extraction → CNN Layers → BiLSTM → Attention → Classification
```

**Pipeline Components**:

1. **Preprocessing**: Silence removal, normalization, resampling
2. **Feature Extraction**: Mel-spectrograms
3. **CNN Block**: Spatial feature learning from spectrograms
4. **BiLSTM**: Temporal pattern recognition
5. **Attention Layer**: Weighted feature aggregation
6. **Classifier**: Emotion prediction

## Repository Structure

```
sawtiya/
├── README.md
├── requirements.txt
├── setup.py
├── LICENSE
├── checkpoint/              # Saved model checkpoints
├── data/
│   └── audio/              # Audio dataset directory
└── src/
    ├── constant/
    │   └── constant.py     # Global configuration
    ├── data/
    │   ├── loader.py       # Metadata extraction
    │   ├── custom_data.py  # PyTorch Dataset class
    │   └── data_loader.py  # DataLoader creation
    ├── augmentation/
    │   └── augmentation.py # Audio augmentation techniques
    ├── model/
    │   └── model.py        # Neural network architecture
    ├── training/
    │   └── train.py        # Training loop and evaluation
    ├── pipeline/
    │   ├── pipeline.py     # End-to-end pipeline
    │   └── main.py         # CLI entry point
    ├── utils/
    │   └── helper.py       # Utility functions
    └── visualization/
        └── plot.py         # Plotting utilities
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for acceleration)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/sawtiya.git
cd sawtiya
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**For GPU Support** (CUDA 11.8 example):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

> **Tip**: Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) for platform-specific installation commands.

## Data Preparation

### Expected Filename Format

Audio files should follow this naming convention:

```
<modality>-<actor_id>-<emotion_code>-<intensity_code>-<statement>-<repetition>.wav
```

**Example**:

```
03-01-05-01-02-01.wav
│  │  │  │  │  └─ Repetition (01 or 02)
│  │  │  │  └──── Statement ID
│  │  │  └─────── Intensity (01: normal, 02: strong)
│  │  └────────── Emotion code (see EMOTION_MAP)
│  └───────────── Actor ID
└──────────────── Modality (01: full-AV, 02: video-only, 03: audio-only)
```

### Emotion Mapping

Default emotion codes (configurable in `constant.py`):

| Code | Emotion |
|----- |---------|
|  0   | Sad     |
|  1   | Angry   |
|  2   | Disgust |
|  3   | Fear    |
|  4   | Happy   |
|  5   | Neutral |

### Directory Setup

```bash
mkdir -p data/audio
# Copy your .wav files into data/audio/
```

> **Important**: Update `BASE_AUDIO_DIR` in `src/constant/constant.py` if using a different directory.

## Usage

### Quick Start: Train the Model

Run the complete pipeline (load data, train, evaluate):

```bash
python -m src.pipeline.main
```

This will:

1. ✅ Load and parse audio metadata
2. ✅ Split data into train/validation/test sets
3. ✅ Create data loaders with augmentation
4. ✅ Initialize the CNN-BiLSTM model
5. ✅ Train for the specified number of epochs
6. ✅ Save the best model checkpoint

### Visualize Audio Samples

Inspect random audio files and their features:

```python
from src.pipeline.pipeline import AudioPipeline
from src.constant.constant import *
from src.utils.helper import collate_audio_batch

# Initialize pipeline
pipeline = AudioPipeline(
    root_dir=BASE_AUDIO_DIR,
    emotion_map=EMOTION_MAP,
    intensity_map=INTENSITY_MAP,
    label_map=EMOTION_LABEL,
    train_split=TRAIN_SPLIT,
    test_split=TEST_SPLIT,
    sample_rate=SAMPLE_RATE,
    silence_db=SILENCE_THRESHOLD_DB,
    batch_size=BATCH_SIZE,
    epoch=EPOCHS,
    collate_fn=collate_audio_batch,
    time_stretch_prob=AUG_TIME_STRETCH_PROB,
    pitch_shift_prob=AUG_PITCH_SHIFT_PROB,
    noise_prob=AUG_NOISE_PROB,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS,
    checkpoint_path=MODEL_CHECKPOINT_PATH,
)

# Visualize a random sample
pipeline.visualize_sample()
```

## Configuration

All hyperparameters are defined in `src/constant/constant.py`:

### Audio Processing

```python
SAMPLE_RATE = 16000              # Target sampling rate (Hz)
SILENCE_THRESHOLD_DB = 30        # Silence trimming threshold in dB
```

### Feature Extraction

```python
N_FFT = 1024                     # FFT window size
HOP_LENGTH = 256                 # Hop length for STFT
N_MELS = 64                      # Number of Mel bands
NUM_MFCC = 40                     # Number of MFCC coefficients
```

### Training Configuration

```python
BATCH_SIZE = 64                  # Batch size
TRAIN_SPLIT = 0.7                # Training set ratio
TEST_SPLIT = 0.2                 # Test set ratio (remaining = validation)
EPOCHS = 50                      # Number of training epochs
LEARNING_RATE = 0.001            # Initial learning rate
```

### Data Augmentation

```python
AUG_TIME_STRETCH_PROB = 0.5      # Probability of time stretching
AUG_PITCH_SHIFT_PROB = 0.5       # Probability of pitch shifting
AUG_NOISE_PROB = 0.3             # Probability of noise injection
```

### Paths

```python
BASE_AUDIO_DIR = "/media/ahmed/Data/Sawtiya/data/audio"    # Dataset directory
MODEL_CHECKPOINT_PATH = "/media/ahmed/Data/Sawtiya/checkpoint/best_model.pth"
```

## Model Details

### Architecture Overview

```python
Input: Mel-spectrogram (n_mels × time_steps)
    ↓
[Conv2D → BatchNorm → ReLU → MaxPool] × N layers
    ↓
Flatten & Reshape
    ↓
BiLSTM (bidirectional temporal modeling)
    ↓
Attention Layer (weighted aggregation)
    ↓
Fully Connected → Softmax
    ↓
Output: Emotion probabilities
```

### Key Components

- **CNN Layers**: Extract spatial patterns from spectrograms
- **BiLSTM**: Capture temporal dependencies in both directions
- **Attention Mechanism**: Learn which time frames are most emotionally significant
- **Dropout**: Regularization to prevent overfitting

## Contributing

Contributions are welcome! Here's how to get started:

1. **Fork the repository**
2. **Create a feature branch**

   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
4. **Test thoroughly**

   ```bash
   python -m pytest tests/
   ```

5. **Commit with clear messages**

   ```bash
   git commit -m "Add amazing feature"
   ```

6. **Push to your fork**

   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and modular

## Performance Tips

- **Use GPU**: Train 10-20× faster with CUDA
- **Batch Size**: Larger batches (32-64) improve GPU utilization
- **Data Augmentation**: Helps prevent overfitting on small datasets
- **Feature Engineering**: Experiment with different n_mels, n_fft values
- **Ensemble Models**: Combine multiple models for better accuracy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

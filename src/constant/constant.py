from pathlib import Path

# =========================
# Paths
# =========================
BASE_AUDIO_DIR = Path(
    "/media/ahmed/Data/Sawtiya/data/audio"
)  # Path to the dataset audio directory
MODEL_CHECKPOINT_PATH = "/media/ahmed/Data/Sawtiya/checkpoint/best_model.pth"  # Path for saving/loading model checkpoint


# =========================
# Label Mappings
# =========================
EMOTION_MAP = {  # Mapping short emotion tags to readable full names
    "SAD": "Sad",
    "ANG": "Angry",
    "DIS": "Disgust",
    "FEA": "Fear",
    "HAP": "Happy",
    "NEU": "Neutral",
}

INTENSITY_MAP = {  # Mapping intensity abbreviations to descriptions
    "LO": "Low intensity",
    "MD": "Medium intensity",
    "HI": "High intensity",
}

EMOTION_LABEL = {  # Final numeric class encoding for training
    "Sad": 0,
    "Angry": 1,
    "Disgust": 2,
    "Fear": 3,
    "Happy": 4,
    "Neutral": 5,
}


# =========================
# Audio Processing Config
# =========================
SAMPLE_RATE = 16000  # Target audio sampling rate
SILENCE_THRESHOLD_DB = 30  # Silence removal threshold in dB
NUM_MFCC = 40  # Number of MFCC features
N_FFT = 1024  # FFT window size
HOP_LENGTH = 256  # Hop length for STFT
N_MELS = 64  # Number of Mel filter banks


# =========================
# DataLoader / Training Config
# =========================
EPOCHS = 50  # Full passes over training data
BATCH_SIZE = 64  # Batch size for DataLoader
TRAIN_SPLIT = 0.7  # Train split percentage
TEST_SPLIT = 0.2  # Test split percentage (rest = validation)


# =========================
# Data Augmentation Probabilities
# =========================
AUG_TIME_STRETCH_PROB = 0.5  # Probability of applying time stretching
AUG_PITCH_SHIFT_PROB = 0.5  # Probability of applying pitch shifting
AUG_NOISE_PROB = 0.3  # Probability of adding background noise

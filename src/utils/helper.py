import torch
import librosa
import numpy as np
from typing import List, Tuple


def collate_audio_batch(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad variable-length audio feature sequences and stack corresponding labels.

    Args:
        batch (List[Tuple[torch.Tensor, int]]):
            List of (features, label) pairs.
            - features shape: (time, feature_dim)
            - label: int class index

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - padded_features: (batch_size, 1, time, feature_dim)
            - labels: (batch_size,)
    """
    feature_list = [item[0] for item in batch]
    label_tensor = torch.tensor([item[1] for item in batch], dtype=torch.long)

    padded_features = torch.nn.utils.rnn.pad_sequence(
        feature_list, batch_first=True
    ).unsqueeze(
        1
    )  # Add channel dimension

    return padded_features, label_tensor


def load_audio_file(path: str, target_sr: int, silence_db: int) -> np.ndarray:
    """
    Load, normalize, and trim silence from an audio file.

    Args:
        path (str): Path to the audio file.
        target_sr (int): Sampling rate to resample the audio.
        silence_db (int): Threshold (in dB) for trimming silence.

    Returns:
        np.ndarray: Normalized, trimmed audio waveform of shape (samples,)
    """
    waveform, _ = librosa.load(path, sr=target_sr, mono=True)
    waveform = librosa.util.normalize(waveform)
    waveform, _ = librosa.effects.trim(waveform, top_db=silence_db)
    return waveform


def extract_mfcc(waveform: np.ndarray, sample_rate: int, n_mfcc: int) -> torch.Tensor:
    """
    Extract MFCC features from an audio waveform.

    Args:
        waveform (np.ndarray): Audio signal.
        sample_rate (int): Sampling rate of the audio.
        n_mfcc (int): Number of MFCC coefficients.

    Returns:
        torch.Tensor: MFCC matrix of shape (time, n_mfcc)
    """
    mfcc_features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=n_mfcc)
    return torch.from_numpy(mfcc_features.T).float()  # time-first


def extract_mel_spectrogram(
    waveform: np.ndarray, sample_rate: int, n_fft: int, hop_length: int, n_mels: int
) -> torch.Tensor:
    """
    Compute Mel-spectrogram in dB scale from audio waveform.

    Args:
        waveform (np.ndarray): Audio signal.
        sample_rate (int): Sampling rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop size between frames.
        n_mels (int): Number of Mel filter banks.

    Returns:
        torch.Tensor: Mel-spectrogram (time, n_mels) in dB.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return torch.tensor(mel_db.T, dtype=torch.float32)

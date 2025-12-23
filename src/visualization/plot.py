import random
import librosa
import numpy as np
import polars as pl
import librosa.display
from typing import Tuple
import matplotlib.pyplot as plt


class AudioVisualizer:
    """
    Utility for visualizing audio features from files listed in a metadata DataFrame.

    Expected DataFrame columns:
        - "file_path": Path to the audio file
        - "emotion": Emotion label
        - "intensity": Intensity label
    """

    def __init__(self, metadata: pl.DataFrame) -> None:
        """
        Initialize the visualizer with audio metadata.

        Args:
            metadata (pl.DataFrame): DataFrame containing file paths and labels.
        """
        self.metadata: pl.DataFrame = metadata

    def _load_random_audio(self) -> Tuple[np.ndarray, int, str, str]:
        """
        Pick a random audio file from the metadata and load it.

        Returns:
            Tuple[np.ndarray, int, str, str]: audio waveform, sample rate, emotion, intensity
        """
        random_idx = random.randrange(self.metadata.height)

        file_path: str = self.metadata["file_path"][random_idx]
        emotion_label: str = self.metadata["emotion"][random_idx]
        intensity_label: str = self.metadata["intensity"][random_idx]

        waveform, sample_rate = librosa.load(file_path, sr=None)

        return waveform, sample_rate, emotion_label, intensity_label

    def plot_random_sample(self) -> None:
        """
        Display waveform, spectrogram, mel-spectrogram, and MFCC of a random audio file.

        Arranges a 2x2 subplot grid:
            1. Waveform
            2. Log-amplitude spectrogram
            3. Mel-spectrogram
            4. MFCC coefficients
        """
        waveform, sample_rate, emotion, intensity = self._load_random_audio()
        plot_title = f"{emotion} ({intensity})"

        plt.figure(figsize=(20, 10))

        # Waveform
        plt.subplot(2, 2, 1)
        librosa.display.waveshow(waveform, sr=sample_rate)
        plt.title(f"Waveform - {plot_title}")

        # Spectrogram
        plt.subplot(2, 2, 2)
        stft_mag = np.abs(librosa.stft(waveform))
        log_spec = librosa.amplitude_to_db(stft_mag, ref=np.max)
        librosa.display.specshow(log_spec, sr=sample_rate, x_axis="time", y_axis="log")
        plt.title(f"Spectrogram - {plot_title}")
        plt.colorbar(format="%+2.0f dB")

        # Mel-Spectrogram
        plt.subplot(2, 2, 3)
        mel_spec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        librosa.display.specshow(mel_db, sr=sample_rate, x_axis="time", y_axis="mel")
        plt.title(f"Mel-Spectrogram - {plot_title}")
        plt.colorbar(format="%+2.0f dB")

        # MFCC
        plt.subplot(2, 2, 4)
        mfcc_coeffs = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=13)
        librosa.display.specshow(mfcc_coeffs, x_axis="time", sr=sample_rate)
        plt.title(f"MFCC - {plot_title}")

        plt.tight_layout()
        plt.show()

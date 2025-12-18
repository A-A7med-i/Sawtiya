import random
import librosa
import numpy as np


class AudioAugmentor:
    """
    Apply probabilistic audio augmentations: time-stretching, pitch-shifting, and Gaussian noise.

    Each augmentation is applied independently according to its probability.

    Attributes:
        sample_rate (int): Audio sampling rate.
        time_stretch_prob (float): Probability to apply time-stretching.
        pitch_shift_prob (float): Probability to apply pitch-shifting.
        noise_prob (float): Probability to add Gaussian noise.
    """

    def __init__(
        self,
        sample_rate: int,
        time_stretch_prob: float,
        pitch_shift_prob: float,
        noise_prob: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.time_stretch_prob = time_stretch_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.noise_prob = noise_prob

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to the input audio waveform.

        Args:
            waveform (np.ndarray): 1-D audio signal.

        Returns:
            np.ndarray: Augmented audio waveform.
        """

        # Time-stretch (speed up or slow down)
        if random.random() < self.time_stretch_prob:
            stretch_rate = random.uniform(0.9, 1.1)
            waveform = librosa.effects.time_stretch(waveform, rate=stretch_rate)

        # Pitch-shift
        if random.random() < self.pitch_shift_prob:
            semitone_shift = random.uniform(-3, 3)
            waveform = librosa.effects.pitch_shift(
                waveform,
                sr=self.sample_rate,
                n_steps=semitone_shift,
            )

        # Gaussian noise injection
        if random.random() < self.noise_prob:
            noise_level = random.uniform(0.001, 0.008)
            noise = np.random.normal(0, noise_level, size=waveform.shape)
            waveform = waveform + noise

        return waveform

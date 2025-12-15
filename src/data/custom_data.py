import torch
from typing import Union, Callable
from torch.utils.data import Dataset
from src.utils.helper import load_audio_file, extract_mel_spectrogram


class AudioMelDataset(Dataset):
    """
    PyTorch Dataset for loading audio, optionally augmenting it,
    and converting it into Mel-spectrogram features for classification.

    Attributes:
        metadata (object): Structure containing info for each audio sample.
        label_map (dict[str, int]): Maps emotion labels to integer indices.
        augmenter (Callable | None): Optional waveform augmentation function.
        sample_rate (int): Audio sampling rate.
        silence_db (int): Threshold (in dB) to trim silence.
        n_fft (int): FFT window size for Mel-spectrogram.
        hop_length (int): Hop length for Mel-spectrogram.
        n_mels (int): Number of Mel filter banks.
    """

    def __init__(
        self,
        metadata: object,
        label_map: dict[str, int],
        augmenter: Union[Callable, None],
        sample_rate: int,
        silence_db: int,
        n_fft: int,
        hop_length: int,
        n_mels: int,
    ) -> None:
        """
        Initialize the dataset.

        Args:
            metadata (object): Metadata containing (actor_id, emotion, intensity, path).
            label_map (dict[str, int]): Mapping from emotion to label index.
            augmenter (Callable | None): Optional audio augmentation pipeline.
            sample_rate (int): Sampling rate for audio.
            silence_db (int): Threshold in dB for trimming silence.
            n_fft (int): FFT window size.
            hop_length (int): Hop length for Mel-spectrogram.
            n_mels (int): Number of Mel filter banks.
        """
        self.metadata = metadata
        self.label_map = label_map
        self.augmenter = augmenter
        self.sample_rate = sample_rate
        self.silence_db = silence_db
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def __len__(self) -> int:
        """
        Total number of audio samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.FloatTensor, int]:
        """
        Retrieve a sample, optionally augment it, and convert to Mel-spectrogram.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple[torch.FloatTensor, int]: Mel-spectrogram tensor and label index.
        """
        actor_id, emotion_label, intensity_label, file_path = self.metadata.row(index)

        waveform = load_audio_file(file_path, self.sample_rate, self.silence_db)

        if self.augmenter:
            waveform = self.augmenter(waveform)

        mel_spec = extract_mel_spectrogram(
            waveform,
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        mel_spec_tensor = torch.FloatTensor(mel_spec)
        label_index = self.label_map[emotion_label]

        return mel_spec_tensor, label_index

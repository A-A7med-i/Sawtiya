from torch.utils.data import DataLoader
from typing import Any, Callable, Tuple
from src.data.custom_data import AudioMelDataset
from src.augmentation.augmentation import AudioAugmentor


class AudioDataLoaderManager:
    """
    Manage splitting audio metadata and creating PyTorch DataLoaders.

    Attributes:
        metadata (Any): Structure containing all audio sample info.
        train_ratio (float): Fraction of data for training.
        test_ratio (float): Fraction of data for testing.
        label_map (dict[str, int]): Mapping from emotion label to index.
        sample_rate (int): Audio sampling rate.
        silence_db (int): Threshold (in dB) for trimming silence.
        batch_size (int): Batch size for DataLoaders.
        collate_fn (Callable): Function to collate variable-length batches.
        n_fft (int): FFT window size for Mel-spectrogram.
        hop_length (int): Hop length for Mel-spectrogram.
        n_mels (int): Number of Mel filter banks.
        train_augmentor (AudioAugmentor | None): Optional training augmentation pipeline.
    """

    def __init__(
        self,
        metadata: Any,
        train_ratio: float,
        test_ratio: float,
        sample_rate: int,
        silence_db: int,
        use_augmentation: bool,
        batch_size: int,
        collate_fn: Callable,
        label_map: dict[str, int],
        time_stretch_prob: float,
        pitch_shift_prob: float,
        noise_prob: float,
        n_fft: int,
        hop_length: int,
        n_mels: int,
    ) -> None:
        """
        Initialize the DataLoader manager.

        Args:
            metadata (Any): Metadata containing audio sample info.
            train_ratio (float): Fraction of data for training.
            test_ratio (float): Fraction of data for testing.
            sample_rate (int): Audio sampling rate.
            silence_db (int): Threshold for silence trimming.
            use_augmentation (bool): Whether to apply augmentation for training.
            batch_size (int): Batch size for DataLoaders.
            collate_fn (Callable): Function to collate batches.
            label_map (dict[str, int]): Mapping from label string to index.
            time_stretch_prob (float): Probability for time-stretch augmentation.
            pitch_shift_prob (float): Probability for pitch-shift augmentation.
            noise_prob (float): Probability for noise augmentation.
            n_fft (int): FFT window size for Mel-spectrogram.
            hop_length (int): Hop length for Mel-spectrogram.
            n_mels (int): Number of Mel filter banks.
        """
        self.metadata = metadata
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.label_map = label_map
        self.sample_rate = sample_rate
        self.silence_db = silence_db
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.train_augmentor = (
            AudioAugmentor(
                sample_rate=self.sample_rate,
                time_stretch_prob=time_stretch_prob,
                pitch_shift_prob=pitch_shift_prob,
                noise_prob=noise_prob,
            )
            if use_augmentation
            else None
        )

    def split_metadata(self) -> Tuple[Any, Any, Any]:
        """
        Split metadata into train, test, and validation sets.

        Returns:
            Tuple[Any, Any, Any]: (train_metadata, test_metadata, val_metadata)
        """
        total_samples = len(self.metadata)
        train_end = int(self.train_ratio * total_samples)
        test_end = train_end + int(self.test_ratio * total_samples)

        train_metadata = self.metadata[:train_end]
        test_metadata = self.metadata[train_end:test_end]
        val_metadata = self.metadata[test_end:]

        return train_metadata, test_metadata, val_metadata

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train, test, and validation sets.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: (train_loader, test_loader, val_loader)
        """
        train_metadata, test_metadata, val_metadata = self.split_metadata()

        train_dataset = AudioMelDataset(
            metadata=train_metadata,
            label_map=self.label_map,
            augmenter=self.train_augmentor,
            sample_rate=self.sample_rate,
            silence_db=self.silence_db,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        test_dataset = AudioMelDataset(
            metadata=test_metadata,
            label_map=self.label_map,
            augmenter=None,
            sample_rate=self.sample_rate,
            silence_db=self.silence_db,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        val_dataset = AudioMelDataset(
            metadata=val_metadata,
            label_map=self.label_map,
            augmenter=None,
            sample_rate=self.sample_rate,
            silence_db=self.silence_db,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        return train_loader, test_loader, val_loader

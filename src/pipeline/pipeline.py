import torch
import polars as pl
import torch.nn as nn
from typing import Callable
from torch.utils.data import DataLoader
from src.training.train import AudioTrainer
from src.visualization.plot import AudioVisualizer
from src.data.data_loader import AudioDataLoaderManager
from src.data.meta_loader import AudioMetadataExtractor
from src.model.model import CNNBiLSTMAttentionClassifier


class AudioPipeline:
    """
    End-to-end audio processing pipeline.

    Handles:
        - Loading metadata from audio files
        - Visualizing random audio samples
        - Creating PyTorch DataLoaders
        - Initializing the model
        - Training and evaluation
    """

    def __init__(
        self,
        root_dir: str,
        emotion_map: dict[str, str],
        intensity_map: dict[str, str],
        label_map: dict[str, int],
        train_split: float,
        test_split: float,
        sample_rate: int,
        silence_db: int,
        batch_size: int,
        epochs: int,
        collate_fn: Callable,
        time_stretch_prob: float,
        pitch_shift_prob: float,
        noise_prob: float,
        n_fft: int,
        hop_length: int,
        n_mels: int,
        checkpoint_path: str,
    ):
        """
        Initialize the audio pipeline.

        Args:
            root_dir (str): Path to audio files.
            emotion_map (dict): Mapping emotion codes to labels.
            intensity_map (dict): Mapping intensity codes to labels.
            label_map (dict): Mapping emotion labels to integer indices.
            train_split (float): Fraction of data for training.
            test_split (float): Fraction of data for testing.
            sample_rate (int): Audio sample rate.
            silence_db (int): Threshold for trimming silence (dB).
            epochs (int): Number of full iterations over the training dataset.
            batch_size (int): Batch size for DataLoader.
            collate_fn (Callable): Collate function for DataLoader.
            time_stretch_prob (float): Probability for time stretching augmentation.
            pitch_shift_prob (float): Probability for pitch shifting augmentation.
            noise_prob (float): Probability for noise injection augmentation.
            n_fft (int): FFT window size for mel-spectrogram.
            hop_length (int): Hop length for mel-spectrogram.
            n_mels (int): Number of mel filter banks.
            checkpoint_path (str): File path to save the best model checkpoint.
        """
        self.root_dir = root_dir
        self.emotion_map = emotion_map
        self.intensity_map = intensity_map
        self.label_map = label_map

        self.train_split = train_split
        self.test_split = test_split

        self.sample_rate = sample_rate
        self.silence_db = silence_db
        self.batch_size = batch_size
        self.epochs = epochs
        self.collate_fn = collate_fn

        self.time_stretch_prob = time_stretch_prob
        self.pitch_shift_prob = pitch_shift_prob
        self.noise_prob = noise_prob

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.checkpoint_path = checkpoint_path

        # Load metadata
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> pl.DataFrame:
        """
        Load metadata from audio filenames.

        Returns:
            pl.DataFrame: Metadata including actor, emotion, intensity, and file path.
        """
        loader = AudioMetadataExtractor(
            data_dir=self.root_dir,
            emotion_map=self.emotion_map,
            intensity_map=self.intensity_map,
        )
        return loader.load_metadata()

    def visualize_sample(self):
        """
        Plot a random audio sample with waveform, spectrogram, mel-spectrogram, and MFCC.
        """
        plotter = AudioVisualizer(self.metadata)
        plotter.plot_random_sample()

    def create_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split metadata and create PyTorch DataLoaders.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: train, test, validation loaders.
        """
        dataloader_manager = AudioDataLoaderManager(
            metadata=self.metadata,
            train_ratio=self.train_split,
            test_ratio=self.test_split,
            label_map=self.label_map,
            sample_rate=self.sample_rate,
            silence_db=self.silence_db,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            use_augmentation=True,
            time_stretch_prob=self.time_stretch_prob,
            pitch_shift_prob=self.pitch_shift_prob,
            noise_prob=self.noise_prob,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )

        train_loader, test_loader, val_loader = dataloader_manager.get_dataloaders()
        return train_loader, test_loader, val_loader

    def initialize_model(self) -> nn.Module:
        """
        Initialize the emotion recognition model.

        Returns:
            nn.Module: PyTorch model instance.
        """
        return CNNBiLSTMAttentionClassifier(num_emotions=len(self.emotion_map))

    def train(self):
        """
        Run full training pipeline including:
            - Creating DataLoaders
            - Model training with checkpointing
            - Plotting loss and accuracy
            - Running inference on test set
        """
        loss_fn = torch.nn.CrossEntropyLoss()
        model = self.initialize_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        train_loader, test_loader, val_loader = self.create_dataloaders()

        trainer = AudioTrainer(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=self.epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            checkpoint_path=self.checkpoint_path,
        )

        history = trainer.train_model()
        trainer.plot_training_history(history)
        results = trainer.run_inference(test_loader)

        print(
            f"Test Accuracy: {results['accuracy']:.2f}% | F1 Score: {results['f1']:.4f}"
        )

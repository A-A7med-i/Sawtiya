import os
import polars as pl
from pathlib import Path
from tqdm.auto import tqdm
from src.constant.constant import *
from typing import Dict, List, Optional, Union


class AudioMetadataExtractor:
    """
    Extracts structured metadata from audio filenames inside a given directory.

    Expected filename format:
        <actor_id>_<text_code>_<emotion_code>_<intensity_code>.wav

    Example:
        "101_IEO_HAP_HI.wav"

    Attributes:
        data_dir (Path): Directory containing the audio files.
        emotion_map (Dict[str, str]): Maps emotion codes to human-readable labels.
        intensity_map (Dict[str, str]): Maps intensity codes to readable labels.
        wav_files (List[str]): List of discovered .wav filenames.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        emotion_map: Dict[str, str],
        intensity_map: Dict[str, str],
    ) -> None:
        """
        Initialize the metadata extractor.

        Args:
            data_dir (str | Path): Root directory containing audio files.
            emotion_map (dict): Mapping from emotion codes to emotion labels.
            intensity_map (dict): Mapping from intensity codes to intensity labels.
        """
        self.data_dir: Path = Path(data_dir)
        self.emotion_map: Dict[str, str] = emotion_map
        self.intensity_map: Dict[str, str] = intensity_map
        self.wav_files: List[str] = self._scan_audio_files()

    def _scan_audio_files(self) -> List[str]:
        """
        Scan the directory for .wav files.

        Returns:
            List[str]: List of audio filenames.
        """
        return [
            entry.name
            for entry in os.scandir(self.data_dir)
            if entry.is_file() and entry.name.lower().endswith(".wav")
        ]

    def _decode_emotion(self, code: str) -> str:
        """
        Convert an emotion code into its readable label.

        Args:
            code (str): Emotion abbreviation (e.g., 'HAP').

        Returns:
            str: Decoded emotion or "Unknown".
        """
        return self.emotion_map.get(code, "Unknown")

    def _decode_intensity(self, code: str) -> str:
        """
        Convert an intensity code into its readable label.

        Args:
            code (str): Intensity abbreviation (e.g., 'HI').

        Returns:
            str: Decoded intensity or "Unspecified".
        """
        return self.intensity_map.get(code, "Unspecified")

    def _parse_filename(self, filename: str) -> Optional[Dict[str, Union[int, str]]]:
        """
        Parse metadata from a single filename.

        Args:
            filename (str): Name of the audio file.

        Returns:
            dict | None: Contains:
                - actor_id (int)
                - emotion (str)
                - intensity (str)
                - file_path (str)
            Returns None if the filename format is invalid.
        """
        try:
            actor_id_str, _, emotion_code, intensity_with_ext = filename.split("_")
            intensity_code = intensity_with_ext.split(".")[0]

            return {
                "actor_id": int(actor_id_str),
                "emotion": self._decode_emotion(emotion_code),
                "intensity": self._decode_intensity(intensity_code),
                "file_path": str(self.data_dir / filename),
            }

        except Exception as exc:
            print(f"Skipping invalid filename '{filename}': {exc}")
            return None

    def load_metadata(self) -> pl.DataFrame:
        """
        Parse metadata for all discovered audio files and return a Polars DataFrame.

        Returns:
            pl.DataFrame: Data with columns:
                - actor_id
                - emotion
                - intensity
                - file_path
        """
        parsed_items = [
            metadata
            for metadata in (
                self._parse_filename(fname)
                for fname in tqdm(self.wav_files, desc="Parsing Metadata")
            )
            if metadata is not None
        ]

        return pl.DataFrame(parsed_items)

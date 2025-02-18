import os
import logging
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import yaml
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class AudioMetadata:
    """Dataclass to store audio metadata."""
    duration: float
    sample_rate: int
    channels: int
    bitrate: Optional[int]
    format: str
    codec: Optional[str]

class AudioProcessor:
    """Handles audio file processing including loading, metadata extraction, and batch processing."""
    
    def __init__(self, config_path: str = "configs/audio_config.yaml"):
        """Initialize the audio processor with configuration.
        
        Args:
            config_path: Path to the audio configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.target_sr = self.config['audio_processing']['sample_rate']
        self._validate_config()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise
            
    def _validate_config(self):
        """Validate the loaded configuration."""
        required_keys = ['sample_rate', 'supported_formats', 'batch_size']
        for key in required_keys:
            if key not in self.config['audio_processing']:
                raise ValueError(f"Missing required config key: {key}")
    
    def load_audio(self, file_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Load an audio file and return the waveform and sample rate.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            Tuple of (waveform array, sample rate)
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        if file_path.suffix[1:] not in self.config['audio_processing']['supported_formats']:
            raise ValueError(f"Unsupported audio format: {file_path.suffix}")
            
        try:
            # Load audio file
            waveform, sr = librosa.load(
                file_path,
                sr=self.target_sr,
                mono=True
            )
            
            # Validate duration
            duration = len(waveform) / sr
            min_dur = self.config['audio_processing']['min_duration']
            max_dur = self.config['audio_processing']['max_duration']
            
            if duration < min_dur:
                raise ValueError(f"Audio duration ({duration:.2f}s) is too short (min: {min_dur}s)")
            if duration > max_dur:
                raise ValueError(f"Audio duration ({duration:.2f}s) is too long (max: {max_dur}s)")
                
            return waveform, sr
            
        except Exception as e:
            self.logger.error(f"Error loading audio file {file_path}: {e}")
            raise
            
    def extract_metadata(self, file_path: Union[str, Path]) -> AudioMetadata:
        """Extract metadata from an audio file.
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            AudioMetadata object containing file metadata
        """
        file_path = Path(file_path)
        try:
            info = sf.info(file_path)
            return AudioMetadata(
                duration=info.duration,
                sample_rate=info.samplerate,
                channels=info.channels,
                bitrate=getattr(info, 'bitrate', None),
                format=info.format,
                codec=getattr(info, 'subtype', None)
            )
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {file_path}: {e}")
            raise
            
    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[Tuple[np.ndarray, int, AudioMetadata]]:
        """Process a batch of audio files in parallel.
        
        Args:
            file_paths: List of paths to audio files
            
        Returns:
            List of tuples containing (waveform, sample_rate, metadata) for each file
        """
        batch_size = self.config['audio_processing']['batch_size']
        num_workers = self.config['audio_processing']['num_workers']
        
        results = []
        failed_files = []
        
        def process_file(file_path: Union[str, Path]) -> Optional[Tuple[np.ndarray, int, AudioMetadata]]:
            try:
                waveform, sr = self.load_audio(file_path)
                metadata = self.extract_metadata(file_path)
                return waveform, sr, metadata
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append((file_path, str(e)))
                return None
        
        # Process files in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            for batch_start in range(0, len(file_paths), batch_size):
                batch = file_paths[batch_start:batch_start + batch_size]
                futures = [executor.submit(process_file, file_path) for file_path in batch]
                
                # Collect results with progress bar
                with tqdm(total=len(batch), desc="Processing audio batch") as pbar:
                    for future in futures:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                        pbar.update(1)
        
        if failed_files:
            self.logger.warning(f"Failed to process {len(failed_files)} files:")
            for file_path, error in failed_files:
                self.logger.warning(f"  {file_path}: {error}")
                
        return results
        
    def save_audio(self, waveform: np.ndarray, file_path: Union[str, Path], sample_rate: Optional[int] = None):
        """Save audio data to a file.
        
        Args:
            waveform: Audio data as numpy array
            file_path: Output file path
            sample_rate: Sample rate (defaults to config sample_rate if not provided)
        """
        if sample_rate is None:
            sample_rate = self.target_sr
            
        file_path = Path(file_path)
        output_format = self.config['audio_processing']['output_format']
        
        # Create output directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            sf.write(
                file_path,
                waveform,
                sample_rate,
                format=output_format,
                subtype='PCM_16'
            )
        except Exception as e:
            self.logger.error(f"Error saving audio to {file_path}: {e}")
            raise

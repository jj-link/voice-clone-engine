import numpy as np
import torch
import torchaudio
import librosa
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import yaml
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .audio_processor import AudioProcessor

@dataclass
class CleaningResult:
    """Dataclass to store audio cleaning results."""
    waveform: np.ndarray
    sample_rate: int
    noise_reduction_db: float
    peak_db: float
    silence_removed_ms: int
    processing_time_ms: int

class AudioCleaner:
    """Handles audio cleaning operations including noise reduction, normalization, and silence removal."""
    
    def __init__(self, config_path: str = "configs/audio_cleaning_config.yaml"):
        """Initialize the audio cleaner with configuration.
        
        Args:
            config_path: Path to the audio cleaning configuration YAML file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() and 
                                 self.config['audio_cleaning']['processing']['use_gpu'] 
                                 else 'cpu')
        self.audio_processor = AudioProcessor()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            raise
            
    def reduce_noise(self, waveform: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, float]:
        """Reduce background noise using spectral gating or adaptive Wiener filtering.
        
        Args:
            waveform: Input audio waveform
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (cleaned waveform, noise reduction in dB)
        """
        config = self.config['audio_cleaning']['noise_reduction']
        
        # Convert to torch tensor for GPU processing if available
        wav_tensor = torch.tensor(waveform, device=self.device)
        
        if config['method'] == 'spectral_gating':
            # Pad the input to ensure output length matches input
            n_fft = config['n_fft']
            pad_length = (n_fft - len(wav_tensor) % n_fft) % n_fft
            wav_tensor = torch.nn.functional.pad(wav_tensor, (0, pad_length))
            
            # Compute STFT
            stft = torch.stft(
                wav_tensor,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length'],
                window=torch.hann_window(config['win_length']).to(self.device),
                return_complex=True
            )
            
            # Estimate noise profile from non-speech segments
            mag = torch.abs(stft)
            noise_profile = torch.mean(mag, dim=1) + config['n_std_thresh'] * torch.std(mag, dim=1)
            noise_profile = noise_profile.unsqueeze(1)
            
            # Apply spectral gating
            mask = (mag > noise_profile).float()
            stft_cleaned = stft * mask
            
            # Inverse STFT
            cleaned = torch.istft(
                stft_cleaned,
                n_fft=config['n_fft'],
                hop_length=config['hop_length'],
                win_length=config['win_length'],
                window=torch.hann_window(config['win_length']).to(self.device)
            )
            
            # Trim to original length
            cleaned = cleaned[:len(waveform)]
            
        elif config['method'] == 'adaptive_wiener':
            # Implement adaptive Wiener filtering if needed
            raise NotImplementedError("Adaptive Wiener filtering not yet implemented")
            
        else:
            raise ValueError(f"Unknown noise reduction method: {config['method']}")
            
        # Calculate noise reduction in dB
        noise_reduction_db = 20 * np.log10(np.std(waveform) / np.std(cleaned.cpu().numpy()))
        
        return cleaned.cpu().numpy(), noise_reduction_db
        
    def normalize_audio(self, waveform: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize audio to target dBFS level.
        
        Args:
            waveform: Input audio waveform
            
        Returns:
            Tuple of (normalized waveform, peak dB)
        """
        config = self.config['audio_cleaning']['normalization']
        
        if config['method'] == 'peak':
            # Peak normalization
            peak = np.max(np.abs(waveform))
            target_peak = 10 ** ((config['target_db'] - config['headroom_db']) / 20)
            normalized = waveform * (target_peak / peak)
            
        elif config['method'] == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(waveform ** 2))
            target_rms = 10 ** (config['target_db'] / 20)
            normalized = waveform * (target_rms / rms)
            
        else:
            raise ValueError(f"Unknown normalization method: {config['method']}")
            
        peak_db = 20 * np.log10(np.max(np.abs(normalized)))
        
        return normalized, peak_db
        
    def remove_silence(self, waveform: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """Remove silence from audio while preserving small pauses.
        
        Args:
            waveform: Input audio waveform
            sample_rate: Sample rate of the audio
            
        Returns:
            Tuple of (silence-removed waveform, milliseconds of silence removed)
        """
        config = self.config['audio_cleaning']['silence_removal']
        
        # Convert parameters to samples
        min_silence_samples = int(config['min_silence_len'] * sample_rate / 1000)
        keep_silence_samples = int(config['keep_silence'] * sample_rate / 1000)
        
        # Calculate energy threshold
        db_thresh = config['silence_thresh']
        amplitude_thresh = 10 ** (db_thresh / 20)
        
        # Calculate frame-wise RMS energy
        hop_length = min_silence_samples // 4  # Use 1/4 of the window for hop
        energy = librosa.feature.rms(
            y=waveform,
            frame_length=min_silence_samples,
            hop_length=hop_length
        )[0]
        
        # Interpolate energy values to match waveform length
        energy_interp = np.interp(
            np.linspace(0, len(energy), len(waveform)),
            np.arange(len(energy)),
            energy
        )
        
        # Create mask for non-silent regions
        mask = energy_interp > amplitude_thresh
        
        # Dilate mask to keep some silence around speech
        if keep_silence_samples > 0:
            kernel = np.ones(keep_silence_samples)
            mask = np.convolve(mask.astype(float), kernel, mode='same') > 0
            
        # Apply mask
        cleaned = waveform[mask]
        
        # Calculate amount of silence removed
        silence_removed_ms = int((len(waveform) - len(cleaned)) * 1000 / sample_rate)
        
        return cleaned, silence_removed_ms
        
    def clean_audio(self, waveform: np.ndarray, sample_rate: int) -> CleaningResult:
        """Apply full audio cleaning pipeline.
        
        Args:
            waveform: Input audio waveform
            sample_rate: Sample rate of the audio
            
        Returns:
            CleaningResult object with cleaned audio and metrics
        """
        import time
        start_time = time.time()
        
        # 1. Noise Reduction
        cleaned, noise_reduction_db = self.reduce_noise(waveform, sample_rate)
        
        # 2. Silence Removal
        cleaned, silence_removed_ms = self.remove_silence(cleaned, sample_rate)
        
        # 3. Normalization
        cleaned, peak_db = self.normalize_audio(cleaned)
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return CleaningResult(
            waveform=cleaned,
            sample_rate=sample_rate,
            noise_reduction_db=noise_reduction_db,
            peak_db=peak_db,
            silence_removed_ms=silence_removed_ms,
            processing_time_ms=processing_time_ms
        )
        
    def process_batch(self, file_paths: List[Union[str, Path]]) -> List[Tuple[Path, CleaningResult]]:
        """Process a batch of audio files.
        
        Args:
            file_paths: List of paths to audio files
            
        Returns:
            List of tuples containing (file path, cleaning result)
        """
        batch_size = self.config['audio_cleaning']['processing']['batch_size']
        n_jobs = max(1, self.config['audio_cleaning']['noise_reduction']['n_jobs'])
        results = []
        failed_files = []
        
        def process_file(file_path: Union[str, Path]) -> Optional[Tuple[Path, CleaningResult]]:
            try:
                file_path = Path(file_path)
                waveform, sr = self.audio_processor.load_audio(file_path)
                result = self.clean_audio(waveform, sr)
                return file_path, result
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                failed_files.append((file_path, str(e)))
                return None
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            for batch_start in range(0, len(file_paths), batch_size):
                batch = file_paths[batch_start:batch_start + batch_size]
                futures.extend([executor.submit(process_file, file_path) for file_path in batch])
                
            # Collect results with progress bar
            with tqdm(total=len(futures), desc="Cleaning audio batch") as pbar:
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
        
    def save_cleaned_audio(self, result: CleaningResult, output_path: Union[str, Path]):
        """Save cleaned audio to file.
        
        Args:
            result: CleaningResult containing the cleaned audio
            output_path: Path to save the cleaned audio
        """
        self.audio_processor.save_audio(
            result.waveform,
            output_path,
            result.sample_rate
        )

import unittest
import os
import numpy as np
from pathlib import Path
import soundfile as sf
import torch
from src.audio_cleaner import AudioCleaner, CleaningResult

class TestAudioCleaner(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.cleaner = AudioCleaner()
        
        # Create test audio with noise
        self.sample_rate = 22050
        self.duration = 2.0  # seconds
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Generate a signal with speech-like components
        speech = (np.sin(2 * np.pi * 440 * t) +  # Fundamental
                 0.5 * np.sin(2 * np.pi * 880 * t) +  # First harmonic
                 0.25 * np.sin(2 * np.pi * 1320 * t))  # Second harmonic
        
        # Add noise
        noise = np.random.normal(0, 0.1, len(speech))
        self.noisy_speech = speech + noise
        
        # Add silence segments
        silence = np.zeros(int(0.5 * self.sample_rate))  # 0.5 second silence
        self.test_audio = np.concatenate([silence, self.noisy_speech, silence])
        
        # Save test file
        self.test_file = self.test_dir / "test_noisy.wav"
        sf.write(self.test_file, self.test_audio, self.sample_rate)
            
    def tearDown(self):
        """Clean up test environment."""
        # Clean up all test files
        if hasattr(self, 'test_files'):
            for file_path in self.test_files:
                if isinstance(file_path, (str, Path)):
                    file_path = Path(file_path)
                    if file_path.exists():
                        file_path.unlink()
        
        # Clean up main test file
        if hasattr(self, 'test_file'):
            if self.test_file.exists():
                self.test_file.unlink()
        
        # Clean up test directory if it exists and is empty
        if hasattr(self, 'test_dir'):
            if self.test_dir.exists():
                try:
                    self.test_dir.rmdir()
                except OSError:
                    # Directory not empty, clean up remaining files
                    for file in self.test_dir.glob('*'):
                        file.unlink()
                    self.test_dir.rmdir()
            
    def test_noise_reduction(self):
        """Test noise reduction functionality."""
        cleaned, noise_db = self.cleaner.reduce_noise(self.test_audio, self.sample_rate)
        
        # Check shape
        self.assertEqual(len(cleaned), len(self.test_audio))
        
        # Verify noise reduction
        self.assertGreater(noise_db, 0)  # Should have reduced noise
        
        # Check if output is not all zeros
        self.assertGreater(np.std(cleaned), 0)
        
    def test_normalization(self):
        """Test audio normalization."""
        normalized, peak_db = self.cleaner.normalize_audio(self.test_audio)
        
        target_db = self.cleaner.config['audio_cleaning']['normalization']['target_db']
        headroom_db = self.cleaner.config['audio_cleaning']['normalization']['headroom_db']
        
        # Check if peak level is within expected range
        self.assertLessEqual(peak_db, target_db)
        self.assertGreaterEqual(peak_db, target_db - headroom_db - 1)  # Allow 1dB tolerance
        
    def test_silence_removal(self):
        """Test silence removal."""
        cleaned, silence_ms = self.cleaner.remove_silence(self.test_audio, self.sample_rate)
        
        # Should have removed approximately 1 second of silence
        expected_silence_ms = 1000  # 2 * 0.5 seconds of silence
        tolerance_ms = 700  # Allow larger tolerance due to windowing effects
        
        self.assertLess(abs(silence_ms - expected_silence_ms), tolerance_ms)
        
        # Output should be shorter than input
        self.assertLess(len(cleaned), len(self.test_audio))
        
        # Verify that we still have some audio content
        self.assertGreater(len(cleaned), 0)
        self.assertGreater(np.std(cleaned), 0)
        
    def test_full_pipeline(self):
        """Test the complete audio cleaning pipeline."""
        result = self.cleaner.clean_audio(self.test_audio, self.sample_rate)
        
        # Verify result type
        self.assertIsInstance(result, CleaningResult)
        
        # Check noise reduction
        self.assertGreater(result.noise_reduction_db, 0)
        
        # Check normalization
        target_db = self.cleaner.config['audio_cleaning']['normalization']['target_db']
        self.assertLessEqual(result.peak_db, target_db)
        
        # Check silence removal
        self.assertGreater(result.silence_removed_ms, 0)
        
        # Check processing time
        self.assertGreater(result.processing_time_ms, 0)
        
    def test_batch_processing(self):
        """Test batch processing of multiple files."""
        # Create multiple test files
        self.test_files = []
        for i in range(3):
            file_path = self.test_dir / f"test_noisy_{i}.wav"
            sf.write(file_path, self.test_audio, self.sample_rate)
            self.test_files.append(file_path)
            
        # Process batch
        results = self.cleaner.process_batch(self.test_files)
        
        # Verify results
        self.assertEqual(len(results), len(self.test_files))
        for file_path, result in results:
            self.assertIsInstance(result, CleaningResult)
            self.assertGreater(result.noise_reduction_db, 0)
            
if __name__ == '__main__':
    unittest.main()

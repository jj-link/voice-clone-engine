import unittest
import os
import numpy as np
from pathlib import Path
import soundfile as sf
from src.audio_processor import AudioProcessor, AudioMetadata

class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path("tests/test_data")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.processor = AudioProcessor()
        
        # Create a test audio file
        self.sample_rate = 22050
        self.duration = 2.0  # seconds
        self.test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, self.duration, int(self.sample_rate * self.duration)))
        
        # Save test files in different formats
        self.test_files = {}
        for fmt in ['wav', 'flac']:
            path = self.test_dir / f"test_audio.{fmt}"
            sf.write(path, self.test_audio, self.sample_rate)
            self.test_files[fmt] = path
            
    def tearDown(self):
        """Clean up test environment."""
        for file_path in self.test_files.values():
            if file_path.exists():
                file_path.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
            
    def test_load_audio(self):
        """Test audio loading functionality."""
        for fmt, file_path in self.test_files.items():
            with self.subTest(format=fmt):
                waveform, sr = self.processor.load_audio(file_path)
                self.assertEqual(sr, self.sample_rate)
                self.assertEqual(len(waveform), len(self.test_audio))
                # Check if waveforms are similar (allowing for small differences due to encoding)
                self.assertTrue(np.allclose(waveform, self.test_audio, atol=1e-4))
                
    def test_extract_metadata(self):
        """Test metadata extraction."""
        for fmt, file_path in self.test_files.items():
            with self.subTest(format=fmt):
                metadata = self.processor.extract_metadata(file_path)
                self.assertIsInstance(metadata, AudioMetadata)
                self.assertEqual(metadata.sample_rate, self.sample_rate)
                self.assertAlmostEqual(metadata.duration, self.duration, places=1)
                self.assertEqual(metadata.channels, 1)
                
    def test_invalid_file(self):
        """Test handling of invalid files."""
        invalid_path = self.test_dir / "nonexistent.wav"
        with self.assertRaises(FileNotFoundError):
            self.processor.load_audio(invalid_path)
            
    def test_batch_processing(self):
        """Test batch processing of multiple files."""
        file_paths = list(self.test_files.values())
        results = self.processor.process_batch(file_paths)
        
        self.assertEqual(len(results), len(file_paths))
        for waveform, sr, metadata in results:
            self.assertEqual(sr, self.sample_rate)
            self.assertTrue(np.allclose(waveform, self.test_audio, atol=1e-4))
            self.assertIsInstance(metadata, AudioMetadata)
            
    def test_save_audio(self):
        """Test audio saving functionality."""
        output_path = self.test_dir / "output_test.wav"
        self.processor.save_audio(self.test_audio, output_path, self.sample_rate)
        
        self.assertTrue(output_path.exists())
        loaded_audio, loaded_sr = sf.read(output_path)
        self.assertEqual(loaded_sr, self.sample_rate)
        self.assertTrue(np.allclose(loaded_audio, self.test_audio, atol=1e-4))
        
        # Clean up
        output_path.unlink()
        
if __name__ == '__main__':
    unittest.main()

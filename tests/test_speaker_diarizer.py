"""Tests for the Speaker Diarization System."""

import unittest
import torch
import torchaudio
import tempfile
import shutil
import json
import numpy as np
from pathlib import Path
import os
from unittest.mock import MagicMock, patch
import urllib.request
import zipfile
import soundfile as sf

class TestSpeakerDiarizer(unittest.TestCase):
    """Test cases for SpeakerDiarizer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Create test config
        cls.config = {
            'model': {
                'name': 'pyannote/speaker-diarization',
                'pretrained_path': 'pyannote/speaker-diarization',
            },
            'processing': {
                'sample_rate': 16000,
                'min_speech_duration': 0.5,    # Minimum duration of speech segment
                'min_silence_duration': 0.5,    # Minimum duration of silence between speakers
                'speech_pad_duration': 0.1,     # Padding around speech segments
            },
            'optimization': {
                'batch_size': 32,
                'use_gpu': False,  # Use CPU for testing
                'gpu_id': 0,
                'fp16': False
            }
        }
        
        # Write config to temp file
        cls.config_path = cls.test_dir / 'diarizer_config.json'
        with open(cls.config_path, 'w') as f:
            json.dump(cls.config, f, indent=4)
            
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
        
    def setUp(self):
        """Set up each test."""
        pass
        
    def generate_test_audio(self, duration=10.0, sample_rate=16000):
        """Generate test audio with multiple speakers and overlaps."""
        # Generate time points
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Create formant frequencies for different speakers
        speaker_formants = [
            [500, 1500, 2500],  # Male speaker
            [600, 1800, 2800],  # Female speaker
            [550, 1650, 2650],  # Third speaker
        ]
        
        # Initialize audio array
        audio = np.zeros_like(t)
        
        # Generate ground truth segments
        segments = []
        current_time = 0.0
        speaker_idx = 0
        
        while current_time < duration - 1.0:  # Leave room for last segment
            # Randomly decide segment duration (0.5 to 2.0 seconds)
            segment_duration = np.random.uniform(0.5, 2.0)
            
            # Ensure we don't exceed total duration
            if current_time + segment_duration > duration:
                segment_duration = duration - current_time
            
            # Calculate sample indices
            start_idx = int(current_time * sample_rate)
            end_idx = int((current_time + segment_duration) * sample_rate)
            
            # Generate speech for this segment using formants
            segment_audio = np.zeros_like(t[start_idx:end_idx])
            formants = speaker_formants[speaker_idx]
            
            # Add formant frequencies with harmonics
            fundamental_freq = 120 if speaker_idx == 0 else 180  # Different pitch for speakers
            for formant in formants:
                # Add fundamental and harmonics
                for harmonic in range(1, 4):
                    freq = formant * harmonic
                    segment_audio += 0.1 * np.sin(2 * np.pi * freq * t[start_idx:end_idx])
            
            # Apply amplitude envelope
            envelope = np.ones_like(segment_audio)
            fade_samples = int(0.05 * sample_rate)  # 50ms fade
            envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
            envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            segment_audio *= envelope
            
            # Add to main audio
            audio[start_idx:end_idx] += segment_audio
            
            # Add segment to ground truth
            segments.append({
                'start': current_time,
                'end': current_time + segment_duration,
                'speaker_id': str(speaker_idx)
            })
            
            # Sometimes add overlapping speech (reduced probability)
            if np.random.random() < 0.2 and current_time < duration - 1.5:  
                overlap_speaker = (speaker_idx + 1) % len(speaker_formants)
                overlap_duration = np.random.uniform(0.4, 0.8)  
                overlap_start = current_time + 0.2  
                
                # Calculate overlap indices
                overlap_start_idx = int(overlap_start * sample_rate)
                overlap_end_idx = int((overlap_start + overlap_duration) * sample_rate)
                
                # Generate overlapping speech
                overlap_audio = np.zeros_like(t[overlap_start_idx:overlap_end_idx])
                overlap_formants = speaker_formants[overlap_speaker]
                
                fundamental_freq = 120 if overlap_speaker == 0 else 180
                for formant in overlap_formants:
                    for harmonic in range(1, 4):
                        freq = formant * harmonic
                        overlap_audio += 0.1 * np.sin(2 * np.pi * freq * t[overlap_start_idx:overlap_end_idx])
                
                # Apply envelope to overlap
                overlap_envelope = np.ones_like(overlap_audio)
                overlap_envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                overlap_envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
                overlap_audio *= overlap_envelope
                
                # Mix with reduced amplitude
                audio[overlap_start_idx:overlap_end_idx] += 0.5 * overlap_audio  
                
                # Add overlapping segment to ground truth
                segments.append({
                    'start': overlap_start,
                    'end': overlap_start + overlap_duration,
                    'speaker_id': str(overlap_speaker),
                    'is_overlap': True
                })
            
            # Move to next segment
            current_time += segment_duration + np.random.uniform(0.2, 0.4)  
            speaker_idx = (speaker_idx + 1) % len(speaker_formants)
        
        # Normalize audio
        max_val = np.max(np.abs(audio))
        if max_val > 0:  
            audio = audio / max_val
        
        return audio, segments

    def test_speech_detection(self):
        """Test speech detection functionality."""
        # Generate test audio
        audio, ground_truth = self.generate_test_audio()
        
        # Save test audio
        test_file = os.path.join(self.test_dir, "test_audio.wav")
        sf.write(test_file, audio, 16000)
        
        # Initialize diarizer
        from src.speaker_diarizer import SpeakerDiarizer
        diarizer = SpeakerDiarizer(self.config_path)
        
        # Detect speech segments
        segments = diarizer.detect_speech(test_file)
        
        # Basic validation
        self.assertGreater(len(segments), 0, "No speech segments detected")
        
        # Verify segment durations
        for segment in segments:
            duration = segment.end - segment.start
            self.assertGreaterEqual(duration, diarizer.min_speech_duration,
                                  "Speech segment shorter than minimum duration")
            
        # Check for reasonable gaps
        for i in range(len(segments) - 1):
            gap = segments[i + 1].start - segments[i].end
            self.assertGreaterEqual(gap, 0, "Overlapping non-overlap segments detected")
            self.assertLessEqual(gap, 1.0, "Unreasonably large gap between segments")

    def test_diarization(self):
        """Test full diarization functionality."""
        # Generate test audio with known speakers
        audio, ground_truth = self.generate_test_audio()
        
        # Save test audio
        test_file = os.path.join(self.test_dir, "test_diarization.wav")
        sf.write(test_file, audio, 16000)
        
        # Initialize diarizer
        from src.speaker_diarizer import SpeakerDiarizer
        diarizer = SpeakerDiarizer(self.config_path)
        
        # Perform diarization
        segments = diarizer.diarize(test_file)
        
        # Basic validation
        self.assertGreater(len(segments), 0, "No segments detected")
        
        # Check speaker labels
        speaker_ids = set(segment.speaker_id for segment in segments)
        self.assertGreaterEqual(len(speaker_ids), 1, "No speakers identified")
        self.assertLessEqual(len(speaker_ids), 3, "Too many speakers identified")
        
        # Verify overlapping segments
        overlap_segments = [s for s in segments if s.is_overlap]
        ground_truth_overlaps = [s for s in ground_truth if s.get('is_overlap', False)]
        
        # Allow for more tolerance in overlap detection
        self.assertLessEqual(
            abs(len(overlap_segments) - len(ground_truth_overlaps)),
            3,  
            "Overlap detection significantly different from ground truth"
        )
        
        # Check temporal consistency
        for i in range(len(segments) - 1):
            if not segments[i].is_overlap and not segments[i + 1].is_overlap:
                self.assertLessEqual(
                    segments[i].end,
                    segments[i + 1].start,
                    "Non-overlapping segments are overlapping"
                )

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        from src.speaker_diarizer import SpeakerDiarizer
        diarizer = SpeakerDiarizer(self.config_path)
        
        # Test very short audio
        audio, _ = self.generate_test_audio(duration=0.3)  # Shorter than min_speech_duration
        test_file = os.path.join(self.test_dir, "short_audio.wav")
        sf.write(test_file, audio, 16000)
        
        segments = diarizer.diarize(test_file)
        self.assertEqual(len(segments), 0, "Should not detect segments in very short audio")
        
        # Test silence
        silent_audio = np.zeros(16000)  # 1 second of silence
        silent_file = os.path.join(self.test_dir, "silence.wav")
        sf.write(silent_file, silent_audio, 16000)
        
        segments = diarizer.diarize(silent_file)
        self.assertEqual(len(segments), 0, "Should not detect segments in silence")
        
        # Test single speaker
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2, 32000))  # 2 seconds of pure tone
        single_speaker_file = os.path.join(self.test_dir, "single_speaker.wav")
        sf.write(single_speaker_file, audio, 16000)
        
        segments = diarizer.diarize(single_speaker_file)
        if segments:
            speaker_ids = set(segment.speaker_id for segment in segments)
            self.assertEqual(len(speaker_ids), 1, "Should detect only one speaker")

if __name__ == '__main__':
    unittest.main()

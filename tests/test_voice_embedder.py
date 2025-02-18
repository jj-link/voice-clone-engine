"""Tests for the Voice Embedding System."""

import unittest
import torch
import torchaudio
import tempfile
import shutil
import yaml
import numpy as np
from unittest.mock import MagicMock, patch
import os
from pathlib import Path

from src.voice_embedder import VoiceEmbedder
from src.gpu_manager import GPUManager

class TestVoiceEmbedder(unittest.TestCase):
    """Test cases for VoiceEmbedder class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directory for test files
        cls.test_dir = Path(tempfile.mkdtemp())
        
        # Set environment variable to disable symlink warning
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        # Create test config
        cls.config = {
            'model': {
                'name': 'ECAPA-TDNN',
                'pretrained_path': 'speechbrain/spkrec-ecapa-voxceleb',
                'embedding_dim': 192
            },
            'processing': {
                'sample_rate': 16000,
                'chunk_duration': 3.0,
                'overlap': 0.5,
                'min_chunks': 2
            },
            'similarity': {
                'cosine_threshold': 0.75,
                'use_mean_pooling': True
            },
            'optimization': {
                'batch_size': 32,
                'use_gpu': False,  # Use CPU for testing
                'gpu_id': 0,
                'fp16': False
            },
            'cache': {
                'enabled': True,
                'max_cache_size': 1000,
                'persistence_path': str(cls.test_dir / 'cache')
            }
        }
        
        # Write test config
        cls.config_path = cls.test_dir / 'test_config.yaml'
        with open(cls.config_path, 'w') as f:
            yaml.dump(cls.config, f)
        
        # Create test audio files
        cls.sample_rate = 16000
        cls.duration = 10  # seconds
        cls.test_audio = torch.randn(1, cls.sample_rate * cls.duration)
        cls.test_audio_path = cls.test_dir / 'test_audio.wav'
        torchaudio.save(cls.test_audio_path, cls.test_audio, cls.sample_rate)
        
        # Create another test audio file with different content
        cls.test_audio2 = torch.randn(1, cls.sample_rate * cls.duration)
        cls.test_audio2_path = cls.test_dir / 'test_audio2.wav'
        torchaudio.save(cls.test_audio2_path, cls.test_audio2, cls.sample_rate)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir)
    
    def setUp(self):
        """Set up each test."""
        self.gpu_manager = GPUManager()
        
        # Create mock model for testing
        self.mock_model = MagicMock()
        self.mock_model.encode_batch.return_value = torch.randn(1, self.config['model']['embedding_dim'])
        
        # Patch EncoderClassifier to return mock model
        with patch('speechbrain.pretrained.EncoderClassifier.from_hparams', 
                  return_value=self.mock_model):
            self.embedder = VoiceEmbedder(str(self.config_path), self.gpu_manager)
    
    def test_initialization(self):
        """Test VoiceEmbedder initialization."""
        self.assertIsNotNone(self.embedder.model)
        self.assertEqual(self.embedder.device.type, 'cpu')  # Using CPU for testing
        self.assertTrue(hasattr(self.embedder, 'embedding_cache'))
    
    def test_audio_chunking(self):
        """Test audio chunking functionality."""
        audio = torch.randn(1, self.sample_rate * 5)  # 5 seconds
        chunks = self.embedder._get_audio_chunks(audio, self.sample_rate)
        
        # Calculate expected number of chunks
        chunk_duration = self.config['processing']['chunk_duration']
        overlap = self.config['processing']['overlap']
        expected_chunks = int((5 - chunk_duration) / (chunk_duration - overlap)) + 1
        
        self.assertEqual(len(chunks), expected_chunks)
        
        # Check chunk size
        chunk_samples = int(chunk_duration * self.sample_rate)
        self.assertEqual(chunks[0].shape[1], chunk_samples)
    
    def test_embedding_extraction(self):
        """Test voice embedding extraction."""
        embedding = self.embedder.extract_embedding(self.test_audio_path)
        
        # Check embedding dimensions
        self.assertEqual(embedding.shape[0], self.config['model']['embedding_dim'])
        
        # Check embedding is normalized
        norm = torch.norm(embedding).item()
        self.assertAlmostEqual(norm, 1.0, places=6)
    
    def test_embedding_cache(self):
        """Test embedding cache functionality."""
        # Extract embedding first time
        embedding1 = self.embedder.extract_embedding(self.test_audio_path)
        
        # Should load from cache second time
        with patch.object(self.embedder.model, 'encode_batch') as mock_encode:
            embedding2 = self.embedder.extract_embedding(self.test_audio_path)
            mock_encode.assert_not_called()
        
        # Embeddings should be identical
        self.assertTrue(torch.allclose(embedding1, embedding2))
    
    def test_similarity_computation(self):
        """Test similarity computation between embeddings."""
        emb1 = self.embedder.extract_embedding(self.test_audio_path)
        emb2 = self.embedder.extract_embedding(self.test_audio2_path)
        
        similarity = self.embedder.compute_similarity(emb1, emb2)
        
        # Check similarity is between -1 and 1
        self.assertGreaterEqual(similarity, -1.0)
        self.assertLessEqual(similarity, 1.0)
    
    def test_same_speaker_detection(self):
        """Test same speaker detection."""
        emb1 = self.embedder.extract_embedding(self.test_audio_path)
        
        # Same embedding should be detected as same speaker
        self.assertTrue(self.embedder.is_same_speaker(emb1, emb1))
        
        # Different random audio should be detected as different speaker
        emb2 = self.embedder.extract_embedding(self.test_audio2_path)
        self.assertFalse(self.embedder.is_same_speaker(emb1, emb2))
    
    def test_cache_management(self):
        """Test cache size management."""
        # Set very small cache size
        self.embedder.config['cache']['max_cache_size'] = 2
        
        # Extract embeddings for multiple files
        for i in range(3):
            audio = torch.randn(1, self.sample_rate * self.duration)
            audio_path = self.test_dir / f'test_audio_{i}.wav'
            torchaudio.save(audio_path, audio, self.sample_rate)
            self.embedder.extract_embedding(audio_path)
        
        # Check cache size hasn't exceeded limit
        self.assertLessEqual(len(self.embedder.embedding_cache), 2)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Extract an embedding to populate cache
        self.embedder.extract_embedding(self.test_audio_path)
        
        # Clear cache
        self.embedder.clear_cache()
        
        # Check cache is empty
        self.assertEqual(len(self.embedder.embedding_cache), 0)
        self.assertEqual(len(list(self.embedder.cache_path.glob("*.pkl"))), 0)

if __name__ == '__main__':
    unittest.main()

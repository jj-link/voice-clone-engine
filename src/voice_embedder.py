"""Voice Embedding System for extracting speaker embeddings from audio.

This module provides functionality to:
1. Extract voice embeddings using pre-trained models
2. Compare voice similarities
3. Cache embeddings for efficient retrieval
4. Optimize processing using GPU acceleration
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple
import yaml
import logging
from tqdm import tqdm
import hashlib
import pickle
from collections import OrderedDict
import os
import tempfile

from speechbrain.pretrained import EncoderClassifier
from .gpu_manager import GPUManager

class VoiceEmbedder:
    """Main class for handling voice embedding operations."""
    
    def __init__(self, config_path: str, gpu_manager: Optional[GPUManager] = None):
        """Initialize the voice embedding system.
        
        Args:
            config_path: Path to the embedding configuration file
            gpu_manager: Optional GPU manager for resource allocation
        """
        self.config = self._load_config(config_path)
        self.gpu_manager = gpu_manager
        self._setup_device()
        self._init_model()
        self._init_cache()
        
        self.logger = logging.getLogger(__name__)
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_device(self):
        """Set up the processing device (CPU/GPU)."""
        if self.config['optimization']['use_gpu']:
            if self.gpu_manager:
                self.device = self.gpu_manager.get_device()
            else:
                self.device = torch.device(f"cuda:{self.config['optimization']['gpu_id']}" 
                                         if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
            
        self.fp16 = self.config['optimization']['fp16'] and self.device.type == 'cuda'
    
    def _init_model(self):
        """Initialize the pre-trained embedding model."""
        # Set environment variable to disable symlink warning and use copy instead
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        os.environ['SPEECHBRAIN_DISABLE_SYMLINKS'] = 'True'
        
        self.model = EncoderClassifier.from_hparams(
            source=self.config['model']['pretrained_path'],
            run_opts={"device": self.device},
            savedir=Path(tempfile.mkdtemp())  # Use temporary directory for model files
        )
        
        if self.fp16:
            self.model = self.model.half()
    
    def _init_cache(self):
        """Initialize the embedding cache."""
        self.cache_enabled = self.config['cache']['enabled']
        if self.cache_enabled:
            self.cache_path = Path(self.config['cache']['persistence_path'])
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.embedding_cache = OrderedDict()
            self._load_cache()
    
    def _load_cache(self):
        """Load cached embeddings from disk."""
        if not self.cache_enabled:
            return
            
        cache_files = list(self.cache_path.glob("*.pkl"))
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    embedding_data = pickle.load(f)
                    self.embedding_cache.update(embedding_data)
            except Exception as e:
                self.logger.warning(f"Failed to load cache file {cache_file}: {e}")
    
    def _get_audio_chunks(self, audio: torch.Tensor, sample_rate: int) -> List[torch.Tensor]:
        """Split audio into overlapping chunks for processing.
        
        Args:
            audio: Input audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(self.config['processing']['chunk_duration'] * sample_rate)
        overlap_samples = int(self.config['processing']['overlap'] * sample_rate)
        stride = chunk_samples - overlap_samples
        
        chunks = []
        for start in range(0, len(audio[0]) - chunk_samples + 1, stride):
            chunk = audio[:, start:start + chunk_samples]
            chunks.append(chunk)
            
        return chunks
    
    def _compute_embedding(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Compute embedding for a single audio chunk.
        
        Args:
            audio: Input audio tensor
            sample_rate: Audio sample rate
            
        Returns:
            Embedding tensor
        """
        if sample_rate != self.config['processing']['sample_rate']:
            audio = torchaudio.transforms.Resample(
                sample_rate, 
                self.config['processing']['sample_rate']
            )(audio)
            
        with torch.amp.autocast('cuda', enabled=self.fp16):
            embedding = self.model.encode_batch(audio)
            
        # Mean pool and normalize
        embedding = embedding.mean(dim=0)
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
        
        return embedding
    
    def extract_embedding(self, audio_path: Union[str, Path]) -> torch.Tensor:
        """Extract voice embedding from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Voice embedding tensor
        """
        # Check cache first
        if self.cache_enabled:
            cache_key = self._get_cache_key(audio_path)
            if cache_key in self.embedding_cache:
                return self.embedding_cache[cache_key]
        
        # Load and process audio
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.to(self.device)
        
        # Split into chunks
        chunks = self._get_audio_chunks(audio, sample_rate)
        if len(chunks) < self.config['processing']['min_chunks']:
            raise ValueError(f"Audio file too short. Need at least "
                           f"{self.config['processing']['chunk_duration'] * self.config['processing']['min_chunks']} "
                           f"seconds of audio.")
        
        # Process chunks in batches
        batch_size = self.config['optimization']['batch_size']
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch = torch.stack(batch)
            embedding = self._compute_embedding(batch, sample_rate)
            embeddings.append(embedding)
        
        # Combine chunk embeddings
        final_embedding = torch.stack(embeddings).mean(dim=0)
        
        # Cache the result
        if self.cache_enabled:
            self._cache_embedding(cache_key, final_embedding)
        
        return final_embedding
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> float:
        """Compute cosine similarity between two voice embeddings.
        
        Args:
            emb1: First embedding tensor
            emb2: Second embedding tensor
            
        Returns:
            Similarity score between 0 and 1
        """
        emb1 = emb1.to(self.device)
        emb2 = emb2.to(self.device)
        
        similarity = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0)
        )
        
        return similarity.item()
    
    def is_same_speaker(self, emb1: torch.Tensor, emb2: torch.Tensor) -> bool:
        """Determine if two embeddings are from the same speaker.
        
        Args:
            emb1: First embedding tensor
            emb2: Second embedding tensor
            
        Returns:
            True if embeddings are likely from the same speaker
        """
        similarity = self.compute_similarity(emb1, emb2)
        return similarity >= self.config['similarity']['cosine_threshold']
    
    def _get_cache_key(self, audio_path: Union[str, Path]) -> str:
        """Generate a cache key for an audio file."""
        audio_path = Path(audio_path)
        stats = audio_path.stat()
        key_data = f"{audio_path.absolute()}{stats.st_size}{stats.st_mtime}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def _cache_embedding(self, key: str, embedding: torch.Tensor):
        """Cache an embedding and manage cache size."""
        if len(self.embedding_cache) >= self.config['cache']['max_cache_size']:
            # Remove oldest item
            self.embedding_cache.popitem(last=False)
        
        self.embedding_cache[key] = embedding
        
        # Persist to disk
        cache_file = self.cache_path / f"{key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump({key: embedding}, f)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache_enabled:
            self.embedding_cache.clear()
            for cache_file in self.cache_path.glob("*.pkl"):
                cache_file.unlink()

"""Voice embedder module for extracting voice embeddings from audio files."""

import os
import torch
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from speechbrain.pretrained import EncoderClassifier
import speechbrain.pretrained.fetching as sb_fetching
import hashlib
import torchaudio
import logging
from collections import OrderedDict
import torch.nn.functional as F
from torch.cuda.amp import autocast

# Monkey patch Path.symlink_to to use copy instead
def _symlink_to(self, target, target_is_directory=False):
    """Copy file instead of creating symlink."""
    import shutil
    shutil.copy2(target, self)

Path.symlink_to = _symlink_to

class VoiceEmbedder:
    """Voice embedder class for extracting voice embeddings from audio files."""
    
    def __init__(self, config_path: str):
        """Initialize voice embedder.
        
        Args:
            config_path: Path to config file
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Set device
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            self.device = 'cuda'
            self.logger.info("Using GPU for voice embedding")
        else:
            self.device = 'cpu'
            self.logger.info("Using CPU for voice embedding")
        
        # Set up half precision if enabled
        self.fp16 = self.config.get('fp16', False)
        if self.fp16 and self.device == 'cuda':
            self.logger.info("Using FP16 precision")
        
        # Initialize cache
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.embedding_cache = {}
        
        # Set up model
        self._setup_model()
        
        # Initialize model with half precision if enabled
        if self.fp16 and self.device == 'cuda':
            self.model = self.model.half()
    
    def extract_embedding(self, audio_input: Union[str, Path, torch.Tensor]) -> torch.Tensor:
        """Extract voice embedding from audio input.
        
        Args:
            audio_input: Audio input, can be:
                - Path to audio file (str or Path)
                - Audio tensor (torch.Tensor)
                
        Returns:
            Voice embedding tensor
        """
        try:
            # Handle audio tensor input
            if isinstance(audio_input, torch.Tensor):
                audio = audio_input
            # Handle file path input
            else:
                audio_path = Path(audio_input)
                if not audio_path.exists():
                    raise FileNotFoundError(f"Audio file not found: {audio_path}")
                    
                # Check cache
                cache_key = self._get_cache_key(audio_path)
                if cache_key in self.embedding_cache:
                    self.logger.info(f"Using cached embedding for {audio_path}")
                    return self.embedding_cache[cache_key]
                
                # Load audio file
                audio = self._load_audio(audio_path)
            
            # Ensure audio has correct shape
            if audio.ndim == 1:
                audio = audio.unsqueeze(0)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.encode_batch(audio)
                embedding = F.normalize(embedding, p=2, dim=1)
            
            # Cache embedding if file path was provided
            if not isinstance(audio_input, torch.Tensor):
                self.embedding_cache[cache_key] = embedding
            
            return embedding
            
        except Exception as e:
            self.logger.error(f"Failed to extract embedding: {e}")
            raise
    
    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load audio file."""
        
        audio, sample_rate = torchaudio.load(audio_path)
        audio = audio.to(self.device)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        return audio
    
    def compute_similarity(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between two embeddings."""
        # Ensure embeddings are 2D
        if emb1.ndim == 1:
            emb1 = emb1.unsqueeze(0)
        if emb2.ndim == 1:
            emb2 = emb2.unsqueeze(0)
            
        # Compute similarity
        sim = F.cosine_similarity(emb1, emb2, dim=1)
        return sim.mean().item()
    
    def _setup_model(self):
        """Set up the embedding model."""
        try:
            # Get model path from config
            model_path = self.config.get('model_path', 'speechbrain/spkrec-ecapa-voxceleb')
            
            # Download and load model
            self.model = EncoderClassifier.from_hparams(
                source=model_path,
                savedir=os.path.expanduser('~/.voice_cloning_ai/models'),
                run_opts={'device': self.device}
            )
            
            self.logger.info(f"Initialized VoiceEmbedder (device: {self.device}, fp16: {self.fp16})")
            
        except Exception as e:
            self.logger.error(f"Failed to set up model: {e}")
            raise
    
    def _get_cache_key(self, audio_path: Union[str, Path]) -> str:
        """Generate cache key for audio file."""
        audio_path = Path(audio_path)
        return hashlib.md5(str(audio_path.absolute()).encode()).hexdigest()
    
    def _cache_embedding(self, cache_key: str, embedding: torch.Tensor):
        """Cache embedding."""
        if self.cache_enabled:
            self.embedding_cache[cache_key] = embedding

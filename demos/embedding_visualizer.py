"""Demo script for visualizing voice embeddings using t-SNE."""

import logging
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchaudio
import tempfile
import urllib.request
from typing import Dict, List
import librosa

from src.voice_embedder import VoiceEmbedder

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingVisualizer:
    """Class for visualizing voice embeddings."""
    
    def __init__(self):
        """Initialize the visualizer with voice embedder."""
        config_path = str(Path(__file__).parent.parent / 'configs' / 'embedding_config.yaml')
        self.embedder = VoiceEmbedder(config_path)
        
        # Set up output directory
        self.output_dir = Path(__file__).parent.parent / 'plots'
        self.output_dir.mkdir(exist_ok=True)
    
    def load_audio(self, audio_path: Path) -> torch.Tensor:
        """Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file (MP3 or WAV)
            
        Returns:
            Preprocessed audio tensor
        """
        # Load audio using librosa (supports MP3)
        audio, sr = librosa.load(str(audio_path), sr=16000, mono=True)
        
        # Convert to torch tensor
        audio_tensor = torch.FloatTensor(audio)
        
        # Ensure proper shape
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        return audio_tensor
    
    def extract_embeddings(self, audio_paths: Dict[str, Path]) -> Dict[str, torch.Tensor]:
        """Extract embeddings from audio files.
        
        Args:
            audio_paths: Dictionary mapping speaker names to audio file paths
            
        Returns:
            Dictionary mapping speaker names to embedding tensors
        """
        embeddings = {}
        for name, path in audio_paths.items():
            logger.info(f"Processing audio for {name}...")
            audio = self.load_audio(path)
            embeddings[name] = self.embedder.extract_embedding(audio)
        return embeddings
    
    def compute_similarity_matrix(self, embeddings: Dict[str, torch.Tensor]) -> np.ndarray:
        """Compute similarity matrix between embeddings.
        
        Args:
            embeddings: Dictionary mapping speaker names to embedding tensors
            
        Returns:
            Similarity matrix as numpy array
        """
        n = len(embeddings)
        sim_matrix = np.zeros((n, n))
        names = list(embeddings.keys())
        
        for i, name1 in enumerate(names):
            for j, name2 in enumerate(names):
                sim_matrix[i, j] = self.embedder.compute_similarity(
                    embeddings[name1], embeddings[name2]
                )
        
        return sim_matrix
    
    def plot_similarity_matrix(self, sim_matrix: np.ndarray, names: List[str]):
        """Plot similarity matrix heatmap."""
        logger.info("Creating similarity matrix plot...")
        try:
            output_path = self.output_dir / 'similarity_matrix.png'
            
            plt.figure(figsize=(10, 8))
            plt.imshow(sim_matrix, cmap='coolwarm', aspect='auto')
            plt.colorbar(label='Similarity')
            
            # Add labels
            plt.xticks(range(len(names)), names, rotation=45)
            plt.yticks(range(len(names)), names)
            
            # Add values in cells
            for i in range(len(names)):
                for j in range(len(names)):
                    plt.text(j, i, f'{sim_matrix[i, j]:.2f}', 
                            ha='center', va='center')
            
            plt.title('Voice Embedding Similarity Matrix')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved similarity matrix plot to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to create similarity matrix plot: {e}")
            raise
    
    def plot_tsne_visualization(self, embeddings: Dict[str, torch.Tensor]):
        """Generate t-SNE visualization of embeddings."""
        logger.info("Creating t-SNE visualization...")
        try:
            output_path = self.output_dir / 'tsne_visualization.png'
            
            # Convert embeddings to numpy array and flatten
            X = torch.stack(list(embeddings.values()), dim=0).cpu().numpy()
            X = X.reshape(len(embeddings), -1)  # Flatten each embedding
            names = list(embeddings.keys())
            
            # Perform t-SNE with lower perplexity
            tsne = TSNE(n_components=2, random_state=42, perplexity=2)
            X_tsne = tsne.fit_transform(X)
            
            # Plot results
            plt.figure(figsize=(10, 8))
            colors = plt.cm.rainbow(np.linspace(0, 1, len(names)))
            
            for i, name in enumerate(names):
                plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=[colors[i]], label=name)
            
            plt.title('t-SNE Visualization of Voice Embeddings')
            plt.xlabel('t-SNE dimension 1')
            plt.ylabel('t-SNE dimension 2')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Saved t-SNE plot to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to create t-SNE plot: {e}")
            raise
    
    def visualize_demo(self):
        """Run the complete visualization demo."""
        logger.info("Starting visualization demo...")
        
        # Get sample audio paths
        logger.info("Looking for sample audio files...")
        samples_dir = Path(__file__).parent.parent / 'samples'
        logger.info(f"Samples directory: {samples_dir}")
        
        if not samples_dir.exists():
            logger.error(f"Samples directory not found: {samples_dir}")
            return
            
        audio_files = list(samples_dir.glob('*.mp3'))
        logger.info(f"Found audio files: {[f.name for f in audio_files]}")
        
        if not audio_files:
            logger.error("No .mp3 files found in samples directory")
            return
            
        audio_paths = {path.stem: path for path in audio_files}
        
        # Extract embeddings
        logger.info("Extracting voice embeddings...")
        try:
            embeddings = self.extract_embeddings(audio_paths)
            logger.info(f"Successfully extracted embeddings for {len(embeddings)} speakers")
        except Exception as e:
            logger.error(f"Failed to extract embeddings: {e}")
            raise
        
        if len(embeddings) < 2:
            logger.error("Need at least 2 samples for visualization")
            return
        
        # Compute similarity matrix
        logger.info("Computing similarity matrix...")
        try:
            sim_matrix = self.compute_similarity_matrix(embeddings)
            logger.info("Successfully computed similarity matrix")
        except Exception as e:
            logger.error(f"Failed to compute similarity matrix: {e}")
            raise
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        try:
            self.plot_similarity_matrix(sim_matrix, list(embeddings.keys()))
            self.plot_tsne_visualization(embeddings)
            logger.info("Successfully generated visualizations")
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {e}")
            raise
        
        logger.info("Demo completed successfully! Check similarity_matrix.png and tsne_visualization.png in the plots directory")

if __name__ == "__main__":
    visualizer = EmbeddingVisualizer()
    visualizer.visualize_demo()

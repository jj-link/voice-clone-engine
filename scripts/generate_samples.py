"""Script to generate sample audio files for testing using gTTS."""

import os
from pathlib import Path
from gtts import gTTS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sample texts with different speakers and content
SAMPLE_TEXTS = {
    "male_speaker1": "Hello, this is a test of the voice embedding system.",
    "male_speaker2": "The quick brown fox jumps over the lazy dog.",
    "female_speaker1": "Testing different voice characteristics and patterns.",
    "female_speaker2": "This is another sample for our voice embedding visualization."
}

def generate_samples():
    """Generate sample audio files using gTTS."""
    samples_dir = Path(__file__).parent.parent / 'samples'
    samples_dir.mkdir(exist_ok=True)
    
    for speaker, text in SAMPLE_TEXTS.items():
        output_path = samples_dir / f"{speaker}.wav"
        
        try:
            # Generate TTS audio directly as MP3 (we'll work with MP3 files for now)
            tts = gTTS(text=text, lang='en', slow=False)
            mp3_path = output_path.with_suffix('.mp3')
            tts.save(str(mp3_path))
            logger.info(f"Generated sample audio for {speaker}")
            
        except Exception as e:
            logger.error(f"Failed to generate sample for {speaker}: {e}")

if __name__ == "__main__":
    generate_samples()

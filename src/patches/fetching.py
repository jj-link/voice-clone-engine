"""Patched version of SpeechBrain's fetching module that uses copy instead of symlinks."""

import os
import shutil
from pathlib import Path
from speechbrain.pretrained.fetching import fetch as original_fetch

def fetch(destination, source, overwrite=False, auto_downsample=False):
    """Modified fetch function that uses copy instead of symlinks.
    
    Args:
        destination: Where to put the fetched file
        source: URI of where to get the file
        overwrite: If True, overwrite the destination
        auto_downsample: If True, automatically downsample audio files
        
    Returns:
        pathlib.Path of the downloaded file
    """
    # Get the file using original fetch
    sourcepath = original_fetch(destination, source, overwrite, auto_downsample)
    
    # If destination exists and is a symlink, remove it
    if destination.exists():
        if destination.is_symlink():
            destination.unlink()
    
    # Create parent directories if they don't exist
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy the file instead of creating a symlink
    shutil.copy2(sourcepath, destination)
    
    return destination

"""Launcher script that patches pathlib for Windows compatibility."""

import os
import shutil
from pathlib import Path

# Monkey patch Path.symlink_to to use copy instead
def safe_symlink_to(self, target, target_is_directory=False):
    try:
        if os.path.exists(self):
            return
        if isinstance(target, Path):
            target = str(target)
        shutil.copy2(target, self)
    except Exception as e:
        print(f"Warning: Failed to copy {target} to {self}: {e}")

Path.symlink_to = safe_symlink_to

# Now import and run the actual demo
import embedding_visualizer

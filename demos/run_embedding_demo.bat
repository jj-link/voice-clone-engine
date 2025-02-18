@echo off
cd %~dp0..
set PYTHONPATH=%CD%
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set SPEECHBRAIN_DISABLE_SYMLINKS=True
python demos\embedding_visualizer.py

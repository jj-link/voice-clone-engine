# Voice Clone Engine

A robust, multi-GPU voice cloning system capable of generating high-quality synthetic voices from audio samples.

## Features

- Multi-GPU support with dynamic resource management
- Advanced GPU selection and monitoring capabilities
- Flexible configuration system
- Comprehensive test coverage

## Requirements

- Python 3.8+
- CUDA-compatible NVIDIA GPU(s)
- PyTorch with CUDA support
- NVIDIA drivers and CUDA toolkit

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd voice-cloning-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The system provides a flexible GPU management system that can be used as follows:

```python
from src.gpu_manager import GPUManager

# Use as context manager
with GPUManager() as gpu_manager:
    # Get available GPU information
    gpu_info = gpu_manager.get_gpu_info()
    
    # Select specific GPUs
    devices = gpu_manager.setup_torch_devices([0, 1])  # Use GPUs 0 and 1
    
    # Or let the system select the best GPUs
    best_gpus = gpu_manager.parse_gpu_args("best2")  # Select 2 best GPUs
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Project Structure

```
voice-cloning-ai/
├── configs/           # Configuration files
├── data/             # Dataset storage (gitignored)
├── models/           # Model checkpoints (gitignored)
├── src/              # Source code
│   └── gpu_manager.py
└── tests/            # Unit tests
```

## License

[Your chosen license]

## Contributing

[Your contribution guidelines]

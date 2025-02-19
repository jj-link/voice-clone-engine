# Voice Clone Engine

A robust, flexible voice cloning system with advanced speaker diarization and configurable resource management.

## Features

- Advanced speaker diarization
- Flexible resource allocation (CPU/GPU)
- Comprehensive test coverage
- Efficient caching system
- Modular pipeline design

## Requirements

- Python 3.13.2
- PyTorch 2.1.0
- torchaudio 2.1.0
- Compatible with both CPU and GPU environments

## Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd voice-clone-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

🚧 This project is currently under development. For development status and planned features, see:
- `REFACTOR_PLAN.md` - Detailed development roadmap
- `REQUIREMENTS.md` - Project specifications and requirements

### Resource Management

The system supports flexible resource allocation:

```python
from src.resource_manager import ResourceManager

with ResourceManager() as rm:
    # Get available resource information
    resource_info = rm.get_resource_info()
    
    # Configure resource usage
    rm.configure("--resource all")                # Use all available resources
    rm.configure("--resource cpu")                # CPU only
    rm.configure("--resource gpu")                # GPU only
    rm.configure("--resource count:3")            # Use any 3 available resources
    rm.configure("--resource-exclude cpu")        # Use all except CPU
    rm.configure("--resource-memory-limit 80")    # Limit memory usage to 80%
    rm.configure("--resource-strategy balanced")  # Distribute load evenly
```

### Running Demos

The project includes several demos:
```bash
cd demos
python embedding_demo.py     # Voice embedding visualization
python diarization_demo.py   # Speaker diarization
python caching_demo.py       # Performance impact of caching
```

## Project Structure

```
voice-clone-engine/
├── configs/           # Configuration files
├── data/             # Dataset storage (gitignored)
├── demos/            # Demo scripts
├── src/              # Source code
│   ├── speaker_diarizer.py
│   └── resource_manager.py
├── tests/            # Unit tests
└── plots/            # Visualization outputs
```

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

For development guidelines and testing practices, see `REFACTOR_PLAN.md`.

## License

[Your chosen license]

## Contributing

See `REFACTOR_PLAN.md` for development guidelines and workflow.

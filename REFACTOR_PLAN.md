# Voice Clone Engine - Refactor Plan

## Overview
This document outlines both completed work and future steps for the Voice Clone Engine project. Each phase is broken down into specific tasks with clear objectives and acceptance criteria.

## Completed Work

### Phase 1: Core Infrastructure

#### 1.1 Audio File Ingestion 
- [x] Implement MP3 file loading
- [x] Add batch processing capability
- [x] Create error handling for corrupt files
- [x] Verify audio metadata extraction
- [x] Handle files of varying lengths

#### 1.2 Audio Cleaning Pipeline 
- [x] Implement background noise reduction
- [x] Add audio normalization
- [x] Create silence removal
- [x] Achieve >10dB noise reduction
- [x] Maintain voice quality after cleaning
- [x] Process 1 hour of audio in under 5 minutes

#### 1.3 GPU Management System 
- [x] Implement dynamic multi-GPU detection
- [x] Add flexible GPU allocation strategies
- [x] Create resource monitoring
- [x] Implement load balancing
- [x] Add topology-aware GPU selection
- [x] Create memory usage limits
- [x] Support various GPU selection modes:
  ```bash
  --gpu all                # Use all available GPUs
  --gpu 0,2,5             # Use specific GPUs by index
  --gpu range:0-3         # Use GPUs 0 through 3
  --gpu count:3           # Use any 3 available GPUs
  --gpu-exclude 1,3       # Use all GPUs except 1 and 3
  --gpu-memory-limit 80   # Limit each GPU's memory usage to 80%
  --gpu-strategy balanced # Distribute load evenly
  ```

#### 1.4 Voice Embedding System 
- [x] Implement ECAPA-TDNN based voice embedding
- [x] Add GPU acceleration support
- [x] Create voice similarity computation
- [x] Achieve >95% speaker verification accuracy
- [x] Enable real-time processing

#### 1.5 Speaker Diarization (Partial) 
- [x] Implement advanced speech detection
- [x] Add speaker embedding and clustering
- [x] Create overlap detection
- [x] Add spectral feature extraction
- [x] Implement dynamic thresholding
- [x] Create comprehensive test suite

## Remaining Work

### Phase 1: Core Infrastructure (Weeks 1-2)

#### 1.6 Complete Speaker Diarization
- [ ] Add RTTM export functionality
- [ ] Create pipeline interface
- [ ] Add confidence scores
- [ ] Implement real-time processing
- [ ] Integrate with audio cleaning

#### 1.7 Dataset Management System
- [ ] Create `DatasetManager` class
  ```python
  class DatasetManager:
      def __init__(self):
          self.root_dir = Path("datasets")
          self.metadata_store = {}
          self.version_info = {}
  ```
- [ ] Implement file organization structure:
  ```
  datasets/
  ├── raw/                 # Original audio files
  ├── processed/           # Cleaned audio
  │   ├── speaker_segments/# Diarized segments
  │   └── normalized/      # Normalized audio
  ├── models/             # Trained models
  └── metadata/           # JSON metadata files
  ```
- [ ] Add metadata tracking system
  - Speaker information
  - Processing history
  - Quality metrics
- [ ] Implement version control for datasets
- [ ] Create caching mechanism for processed files
- [ ] Add data validation and integrity checks

### Phase 2: Text-to-Speech Core (Weeks 3-4)

#### 2.1 Text Processing
- [ ] Create `TextProcessor` class:
  ```python
  class TextProcessor:
      def normalize_text(self, text: str) -> str
      def convert_to_phonemes(self, text: str) -> List[str]
      def add_prosody_markers(self, phonemes: List[str]) -> List[str]
  ```
- [ ] Implement text normalization
- [ ] Add phoneme conversion
- [ ] Create prosody marking system

#### 2.2 Speech Synthesis
- [ ] Create `SpeechSynthesizer` class:
  ```python
  class SpeechSynthesizer:
      def generate_mel_spectrograms(self, phonemes: List[str]) -> torch.Tensor
      def vocoder_inference(self, mel_specs: torch.Tensor) -> torch.Tensor
      def apply_voice_embedding(self, audio: torch.Tensor, embedding: torch.Tensor)
  ```
- [ ] Implement mel spectrogram generation
- [ ] Add vocoder system
- [ ] Integrate voice embedding
- [ ] Optimize GPU usage

#### 2.3 Runtime Optimization
- [ ] Implement batch processing
- [ ] Add GPU memory management
- [ ] Create pipeline parallelization
- [ ] Optimize data transfers

### Phase 3: Voice Enhancement (Week 5)

#### 3.1 Voice Naturalization
- [ ] Implement prosody modeling
- [ ] Add emotion preservation
- [ ] Create natural pause insertion
- [ ] Develop emphasis system

#### 3.2 Audio Post-processing
- [ ] Create audio smoothing system
- [ ] Implement quality enhancement
- [ ] Add format conversion
- [ ] Create volume normalization

### Phase 4: User Interface (Week 6)

#### 4.1 Command Line Interface
- [ ] Design command structure:
  ```bash
  voice-clone train --input <dir> --model <name> --gpu <ids>
  voice-clone generate --text "Hello" --voice <model> --output <file>
  voice-clone process --input <file> --output <dir>
  voice-clone evaluate --model <name> --test-data <dir>
  ```
- [ ] Implement core commands
- [ ] Add configuration management
- [ ] Create help documentation
- [ ] Implement error handling

#### 4.2 Integration
- [ ] Connect all components
- [ ] Create pipeline manager
- [ ] Implement logging system
- [ ] Add progress tracking

### Phase 5: Testing & Documentation (Week 7)

#### 5.1 Testing
- [ ] Create unit tests for each component
- [ ] Implement integration tests
- [ ] Add performance benchmarks
- [ ] Create quality metrics

#### 5.2 Documentation
- [ ] Write API documentation
- [ ] Create usage examples
- [ ] Add architecture diagrams
- [ ] Write deployment guide

## Development Guidelines

### Test-Driven Development (TDD)
For each new feature or modification:
1. Write failing tests first
   - Define expected behavior
   - Cover edge cases
   - Consider performance requirements
2. Implement minimum code to pass tests
3. Refactor while keeping tests green
4. Update documentation

### Version Control Practices
After each meaningful change:
1. Run all tests to ensure nothing broke
2. Stage changes:
   ```bash
   git add <modified_files>
   ```
3. Create descriptive commit:
   ```bash
   git commit -m "Category: Detailed description

   - Bullet points for specific changes
   - Include any breaking changes
   - Note any performance impacts"
   ```
4. Push to remote:
   ```bash
   git push origin main
   ```

### Code Review Checklist
Before committing:
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Code follows project style
- [ ] No debug code left
- [ ] Performance impact considered

### Commit Categories
Use these prefixes for commits:
- feat: New feature
- fix: Bug fix
- test: Adding/updating tests
- docs: Documentation updates
- perf: Performance improvements
- refactor: Code restructuring
- style: Formatting, missing semicolons, etc.
- chore: Updating build tasks, package manager configs, etc.

Example commit:
```bash
git commit -m "feat: Add real-time speaker diarization

- Implement streaming audio processing
- Add confidence score calculation
- Create pipeline interface
- Update tests for real-time scenarios
- Performance: processes 1hr audio in <5min"
```

## Acceptance Criteria

### Performance
- Process 1 hour of audio in < 5 minutes
- Generate speech in < 3s per sentence
- GPU memory usage < 80%
- CPU usage < 60%

### Quality
- Speaker identification accuracy > 95%
- Voice similarity score > 0.85
- Human evaluation score > 4/5
- Clear pronunciation
- Natural-sounding speech

### Usability
- Clear error messages
- Comprehensive help documentation
- Intuitive command structure
- Proper progress feedback

## Progress Tracking

For each task:
1. Create feature branch
2. Implement functionality
3. Write tests
4. Create PR
5. Review & merge
6. Update documentation

## Notes
- Prioritize GPU optimization throughout
- Maintain backward compatibility
- Focus on code readability
- Keep documentation updated
- Regular performance testing
- Follow TDD principles
- Commit early and often
- Write descriptive commit messages
- Push changes after each feature completion

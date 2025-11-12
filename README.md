
### Modular Automatic Speech Recognition Toolkit

Comprehensive TensorFlow 2 toolkit for building, training, and deploying automatic speech recognition systems.
The project packages DeepSpeech-style acoustic models, configurable pipelines, and utilities that streamline
feature extraction, augmentation, decoding, evaluation, and experiment tracking.

---

#### Key Features
- Modular CTC pipeline that cleanly separates data loading, feature engineering, neural modeling, optimization, and decoding.
- Ready-to-use DeepSpeech and DeepSpeech2 architectures with support for mixed precision and distributed training strategies.
- Audio dataset utilities for WAV ingestion, on-the-fly augmentation, and filter-bank feature computation.
- Decoder implementations for greedy transcription with hooks for integrating custom language models.
- Evaluation helpers for computing character error rate (CER) and word error rate (WER).
- Example scripts and tests covering augmentation, callbacks, datasets, and inference.

---

#### Project Structure
- `automatic_speech_recognition/augmentation` — audio perturbation routines, including SpecAugment.
- `automatic_speech_recognition/callback` — training callbacks such as batch logging and distributed checkpoints.
- `automatic_speech_recognition/dataset` — dataset loaders, audio readers, and feature assembly.
- `automatic_speech_recognition/features` — spectral feature extraction and filter bank implementations.
- `automatic_speech_recognition/model` — DeepSpeech-family architectures built with TensorFlow 2.
- `automatic_speech_recognition/pipeline` — end-to-end CTC pipeline orchestration.
- `automatic_speech_recognition/evaluate` — utilities for activation inspection and error-rate computation.
- `examples/` — runnable scripts illustrating inference, augmentation, and callback usage.
- `tests/` — pytest suite spanning every subsystem to ensure correctness.

---

#### Installation
**Using pip**
```bash
pip install automatic-speech-recognition
```

**From source**
```bash
git clone <repository-url>
cd Automatic-Speech-Recognition
conda env create -f environment.yml      # or use environment-gpu.yml for CUDA-enabled setups
conda activate Automatic-Speech-Recognition
pip install -e .
```

---

#### Quickstart: Inference
```python
import automatic_speech_recognition as asr

sample = asr.utils.read_audio("path/to/audio.wav")
pipeline = asr.load("deepspeech2", lang="en")
transcript = pipeline.predict([sample])
print(transcript[0])
```

---

#### Training Workflow
```python
import numpy as np
import tensorflow as tf
import automatic_speech_recognition as asr

train = asr.dataset.Audio.from_csv("train.csv", batch_size=32)
dev = asr.dataset.Audio.from_csv("dev.csv", batch_size=32)
alphabet = asr.text.Alphabet(lang="en")

features = asr.features.FilterBanks(
    features_num=160,
    winlen=0.02,
    winstep=0.01,
    winfunc=np.hanning,
)

model = asr.model.get_deepspeech2(
    input_dim=160,
    output_dim=alphabet.size,
    rnn_units=800,
    is_mixed_precision=False,
)

optimizer = tf.optimizers.Adam(1e-4)
decoder = asr.decoder.GreedyDecoder()

pipeline = asr.pipeline.CTCPipeline(
    alphabet=alphabet,
    features_extractor=features,
    model=model,
    optimizer=optimizer,
    decoder=decoder,
)

pipeline.fit(train, dev, epochs=25)
pipeline.save("checkpoints/deepspeech2")
```

---

#### Data Preparation
1. Gather 16 kHz, 16-bit mono WAV files.
2. Create CSV manifests with columns: `path`, `transcript`, `duration`.
3. Use `asr.dataset.Audio.from_csv` to construct training, validation, and evaluation datasets.
4. Optionally enable augmentation via `automatic_speech_recognition.augmentation`.

---

#### Evaluation
```python
import automatic_speech_recognition as asr

pipeline = asr.pipeline.CTCPipeline.load("checkpoints/deepspeech2")
dataset = asr.dataset.Audio.from_csv("test.csv")
wer, cer = asr.evaluate.calculate_error_rates(pipeline, dataset)
print(f"WER: {wer:.2%} | CER: {cer:.2%}")
```

Use the activation inspection helpers under `automatic_speech_recognition/evaluate` to analyze intermediate layer outputs.

---

#### Configuration Tips
- Adjust feature extraction parameters in `FilterBanks` to match dataset characteristics.
- Enable mixed precision by setting `is_mixed_precision=True` in model builders when running on GPUs with Tensor Cores.
- Leverage the callbacks in `automatic_speech_recognition/callback` for logging and checkpointing in multi-GPU environments.
- Environment setup recipes for CPU and GPU are provided in `environment.yml` and `environment-gpu.yml`.

---

#### Testing
Run the full test suite after modifying pipeline components or implementing new features.
```bash
pytest
```

---

#### Contributing
- Fork or branch locally, keeping feature scope focused.
- Run formatter and linter checks alongside the pytest suite.
- Include documentation updates in `README.md` or module docstrings for new features.
- Submit concise pull requests describing motivation, implementation details, and testing evidence.

---

#### License
The project is distributed under the terms outlined in `LICENSE.md`.

# Probabilistic Latent Token Generation in GPT-SoVITS

This repository contains the source code, preprocessing pipeline, and experimental framework for an Honors Research Project investigating **Probabilistic Latent Token Generation**. This research transitions the GPT-SoVITS architecture from discrete token selection to a continuous latent space mixture to accelerate acoustic convergence in low-resource settings.

---

## Project Structure

```text
FYP/
├── GPT-SoVITS-v3lora-20250228/  # Modified Model Architecture
├── demo/                        # Reference Audio and Synthesised Audios from Honours Porject Live Demo                        
├── samples/                     # Audio Samples (Reference vs. AI Generated)
│   ├── HiFi/                    # Male/Accented samples
│   └── LJ Speech/               # Female/Neutral samples
├── scripts/                     # Preprocessing & Automation Suite
│   ├── HiFi_TTS_list_files/     # Manifest storage for HPC
│   ├── LJ_Speech_list_files/    # Manifest storage for HPC
│   ├── results/                 # Raw quantitative JSON logs
│   ├── ASR_Replacement_HiFi.py
│   ├── ASR_Replacement_LJ_Speech.py
│   ├── HiFi_TTS_preprocessing.py
│   ├── LJ_Speech_preprocessing.py
│   └── LJ_Speech_preprocessing2.py
├── README.md
└── .gitignore
```

## Technical Contributions

The primary contribution of this project is the re-engineering of the GPT-SoVITS "Text-to-Semantic" and "Semantic-to-Waveform" interface.

### 1. End-to-End Data Pipelines

* **Automated Audio Alignment (ASR Bypass):** Engineered custom data pipeline scripts to handle raw audio transcription, resampling, and strict audio-to-text temporal alignment, effectively bypassing the default Automatic Speech Recognition (ASR) constraints.
* **Dataset Manifest Generation:** Developed robust preprocessing scripts to automatically parse transcribed datasets and generate the highly specific `.list` metadata formats strictly required by the GPT-SoVITS training pipeline, streamlining bulk data ingestion.

### 2. Core Architectural Modifications

* **Autoregressive Probability Extraction (`t2s_model.py`):** Modified the GPT module's inference loop to capture and expose the full unnormalized probability distribution at every timestep (`probs_seq`), preventing the loss of acoustic variance caused by instantaneous discrete multinomial sampling.
* **Continuous Soft Codebook Decoding (`core_vq.py`):** Replaced deterministic, discrete dictionary lookups with a dynamic, probability-weighted mixture of acoustic embeddings. This mathematically preserves 100% of the predicted acoustic distribution:
  $$\tilde{z} = \sum_{i} p_i e_i$$
* **Latent Space Projection Adapter (`models.py`):** Engineered a custom, trainable linear projection layer (`prob_adapter`) to bridge the new high-dimensional continuous soft-token vector ($\tilde{z}$) with the downstream VITS-based acoustic generator, effectively preventing latent space collapse.
* **Stochastic Label Smoothing (`s2_train.py`):** Injected a tunable $\epsilon$ (5%) probability mass distribution during the training phase. This simulates the continuous probabilistic distributions of the inference phase, forcing the linear adapter to learn dense representations rather than overfitting to rigid "one-hot" ground truths.

### 3. Internal Data Verification & Validation

* **End-to-End Differentiability Validation:** Integrated a custom `_debug_prob_step` tensor hook directly inside the `SynthesizerTrn` forward pass to monitor real-time tensor flow across architectural boundaries.
* **Acoustic Padding Verification:** Validated the probabilistic tokenisation logic by empirically proving the model outputs a perfectly uniform distribution ($1/20480$) in zero-gradient silent padding regions, confirming the expected maximum entropy where no acoustic signal is present.

## Audio Samples

A representative selection of synthesised audio is available in the `/samples` directory. These files demonstrate the "early-peaking" advantage of the probabilistic architecture compared to the discrete baseline at low-epoch counts.

* **Reference**: The original human ground truth.
* **Discrete**: Audio generated using the standard deterministic pipeline.
* **Probabilistic**: Audio generated using the proposed continuous soft-token pipeline.

## Configuration & Execution

### Toggling Architectures
The model mode is controlled via the configuration JSON (e.g., `configs/s2.json`):
* `"probabilistic": true`: Enables soft-token mixtures and the Linear Adapter.
* `"probabilistic": false`: Reverts to the baseline discrete lookup.

### Preprocessing Pipeline
The `scripts/` folder contains standalone tools to prepare data and bypass ASR:

```bash
# Example: Generate training manifests with Linux-compatible paths for HPC
python scripts/ASR_Replacement_HiFi.py
```

### Quantitative Evaluation
To calculate the objective metrics (MCD, STOI, PESQ, and ECAPA-TDNN), use the included evaluation script. 

This script requires two separate directories to perform a direct comparison:
1. **Reference Directory (`--ref_dir`)**: A folder containing the original, ground-truth human audio samples.
2. **Generated Directory (`--gen_dir`)**: A folder containing the AI-synthesised audio generated by the model.

```bash
# Example: Running the evaluation suite for the 3-hour discrete model
python tests/evaluation2.py \
  --ref_dir "tests/hifi 3hr tests/discrete/Real" \
  --gen_dir "tests/hifi 3hr tests/discrete/AI" \
  --sr 16000 \
  --compute_eer_random \
  --output "results_disc_hifi_3hr.json"
```

## Prerequisites

In addition to the base [GPT-SoVITS requirements](https://github.com/RVC-Boss/GPT-SoVITS), the following libraries are required to run the custom preprocessing and quantitative evaluation suites:

```bash
pip install numpy scipy soundfile datasets librosa pystoi pesq speechbrain fastdtw scikit-learn
```

## Academic Disclaimer & Attribution

* **Institutional Affiliation:** This project was submitted as part of the BSc (Hons) Computing Science requirements at Heriot-Watt University.
* **Base Code:** Modified from the [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) repository (MIT License).
* **Datasets:** Utilises **LJ Speech** (Public Domain) and **HiFi-TTS** (CC BY 4.0).
* **Ethics:** This repository is for academic research only. Model checkpoints are restricted to prevent unauthorised voice cloning or misuse.

# EquiME: Equitable Micro-Expression Dataset for Cross-Demographic Emotion Recognition

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.xxxxx-b31b1b.svg)](https://arxiv.org/abs/)

---

**EquiME**, a large-scale synthetic dataset for micro-expression analysis, generated using the image-to-video model. By leveraging a structured causal modeling approach, we employ Facial Action Units (AUs) as intermediate representations that drive the generation of realistic ME sequences. This paper presents a streamlined pipeline for generating synthetic micro-expression datasets, designed to be accessible to users without a computer science background. 

---

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Technical Specifications](#technical-specifications)
- [Dataset Organization](#dataset-organization)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Usage Guidelines](#usage-guidelines)
- [Evaluation Protocols](#evaluation-protocols)
- [Ethical Considerations](#ethical-considerations)
- [Licensing](#licensing)
- [Access Policy](#access-policy)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact Information](#contact-information)

---

## Dataset Overview

The ME-VLM dataset addresses critical limitations in micro-expression research by providing a large-scale, controlled synthetic dataset. Micro-expressions, characterized by brief and subtle facial movements lasting 1/25 to 1/5 of a second, are notoriously difficult to capture and annotate in real-world scenarios. Our synthetic approach enables:

- **Scalability**: Generation of large-scale datasets with consistent quality
- **Controllability**: Precise manipulation of emotional expressions and demographic attributes
- **Reproducibility**: Standardized evaluation protocols for fair model comparison
- **Ethical Compliance**: Elimination of privacy concerns associated with real human subjects

### Key Features

- **Scale**: 100,000 high-resolution video sequences
- **Diversity**: Comprehensive coverage of five primary emotional categories
- **Quality**: 512×512 pixel resolution with 24 FPS temporal consistency
- **Annotation**: Rich metadata including demographic attributes and Action Unit annotations
- **Multimodality**: Synchronized video-text pairs for vision-language model training

---

## Technical Specifications

| Parameter | Specification |
|-----------|---------------|
| **Total Samples** | 75,000 video sequences |
| **Video Format** | MP4 (H.264 codec) |
| **Resolution** | 512 × 512 pixels |
| **Frame Rate** | 24 FPS |
| **Duration** | 6.0 seconds per clip |
| **Total Frames** | 144 frames per sequence |
| **Emotion Classes** | 5 categories (happiness, sadness, surprise, fear, anger) |
| **Annotation Format** | JSON metadata files |
| **Source Images** | CelebA-HQ dataset |
| **Generation Model** | LTX-Video diffusion model |

---

## Dataset Organization

The dataset follows a hierarchical structure optimized for research applications:

```
ME-VLM/
├── emotion_categories/
│   ├── happiness/
│   │   ├── [gender]_[subject_id]_[sequence].mp4
│   │   └── metadata/
│   │       └── [video_name]_metadata.json
│   ├── sadness/
│   ├── surprise/
│   ├── fear/
│   └── anger/
├── documentation/
│   ├── generation_protocol.md
│   ├── evaluation_guidelines.md
│   └── annotation_schema.json
└── statistics/
    ├── demographic_distribution.csv
    ├── quality_metrics.csv
    └── dataset_summary.json
```

### Naming Convention

Video files follow the systematic naming pattern:
```
[gender]_[subject_id]_[sequence_number].mp4
```

Where:
- `gender`: Demographic identifier (m/f)
- `subject_id`: Unique subject identifier from source dataset
- `sequence_number`: Sequential identifier for multiple generations per subject

### Metadata Structure

Each video is accompanied by comprehensive metadata including:

- **Demographic Attributes**: Gender, estimated age, ethnicity (via DeepFace analysis)
- **Technical Metadata**: File size, bitrate, duration, frame count
- **Generation Parameters**: Model configuration, prompt templates, random seeds
- **Quality Metrics**: PSNR, SSIM, perceptual quality scores
- **Action Unit Annotations**: Facial muscle movement indicators

---

## Methodology

### Data Generation Pipeline

1. **Source Data Curation**: High-quality facial images selected from CelebA-HQ dataset based on quality metrics and demographic diversity
2. **Prompt Engineering**: Emotion-specific prompts designed to elicit target micro-expressions
3. **Video Synthesis**: LTX-Video model generates temporally consistent sequences
4. **Quality Assessment**: Automated filtering based on perceptual quality metrics
5. **Attribute Extraction**: DeepFace analysis for demographic and emotional attributes
6. **Validation**: Human expert review of subset for quality assurance

### Generation Prompts

Emotion-specific prompts were carefully crafted to ensure consistent and realistic micro-expression generation:

- **Happiness**: "A professional static headshot showing subtle happiness micro-expression transitioning to neutral..."
- **Sadness**: "A professional static headshot showing subtle sadness micro-expression with slight lip corners turning down..."
- **Surprise**: "A professional static headshot showing brief surprise micro-expression with raised eyebrows..."
- **Fear**: "A professional static headshot showing subtle fear micro-expression with widened eyes..."
- **Anger**: "A professional static headshot showing controlled anger micro-expression with tightened facial muscles..."

---

## Repository Structure

### `/generation_pipeline/`
**Purpose**: Complete implementation of the synthetic data generation workflow

**Components**:
- `inference_cropped.py`: Core inference script implementing the LTX-Video generation pipeline
- Handles image preprocessing, model inference, and post-processing
- Supports batch processing for large-scale dataset creation

### `/evaluation_framework/`
**Purpose**: Comprehensive evaluation protocols using established computer vision metrics

**Components**:
- `video_quality_metrics.py`: Implementation of standard video quality assessment metrics
- Evaluation datasets and benchmark results:
  - `summary_metrics_generated5k_mmewAU.csv`: Results on synthetic dataset subset
  - `summary_metrics_miex.csv`: MIEX dataset evaluation
  - `summary_metrics_samm.csv`: SAMM dataset evaluation  
  - `summary_metrics_smic.csv`: SMIC dataset evaluation
  - `video_metrics_hq_emotion.csv`: High-quality emotion-specific metrics

**Supported Metrics**:
- Peak Signal-to-Noise Ratio (PSNR)
- Structural Similarity Index (SSIM)
- Total Variation (TV)
- Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE)
- CLIP-based Image Quality Assessment (CLIP-IQA)

### `/baselines/`
**Purpose**: Reference implementations and benchmark models for comparative evaluation

**Training Scripts**:
- `train_mevlm.py`: ME-VLM model training with 3-class emotion classification
- `train_miex.py`: MIEX dataset baseline implementation
- `train_real.py`: Real data baseline for comparison studies

**Configuration Management**:
- `config/config_3class_mobilenet.json`: Lightweight MobileNet architecture configuration
- `config/config_original.json`: Original baseline model parameters

**Model Architectures**:
- 3D Convolutional Neural Networks for spatiotemporal feature extraction
- ResNet-based architectures with residual connections
- MobileNet variants for efficient inference

### `/attribute_extraction/`
**Purpose**: Demographic and facial attribute analysis tools

**Components**:
- `deepface_only_analysis.py`: Comprehensive facial analysis using the DeepFace framework
- Supports multi-attribute extraction: age, gender, ethnicity, emotional state
- Batch processing capabilities for large-scale analysis
- Configurable backends for optimal performance

### `/static/`
**Purpose**: Web-based documentation and visualization resources

**Structure**:
- `css/`: Stylesheet definitions for web interface
- `images/`: Visual documentation assets
- `js/`: Interactive components and data visualization
- `pdfs/`: Comprehensive documentation in portable format
- `videos/`: Representative sample videos and demonstrations

---

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA GPU with ≥8GB VRAM (A100 or equivalent)
- **RAM**: ≥32GB system memory
- **Storage**: ≥500GB available space for full dataset
- **CPU**: Multi-core processor (≥8 cores recommended)

### Software Dependencies
- **Python**: 3.8 or higher
- **CUDA**: 11.0 or compatible version
- **Operating System**: Linux (Ubuntu 20.04+), Windows 10+, or macOS 11+

---

## Installation

### Environment Setup

1. **Clone the repository**:
```bash
git clone https://github.com/[kirito-blade]/me-vlm.git
cd me-vlm
```

2. **Create virtual environment**:
```bash
python -m venv me_vlm_env
source me_vlm_env/bin/activate  # Linux/macOS
# or
me_vlm_env\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify CUDA installation**:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Usage Guidelines

### Dataset Access
The dataset is available for academic research purposes following proper access approval. Researchers must:

1. Submit formal access request to corresponding author
2. Provide institutional verification
3. Agree to ethical usage terms
4. Cite the work in resulting publications

### Research Applications

**Recommended Use Cases**:
- Micro-expression recognition algorithm development
- Synthetic data augmentation for real datasets
- Multimodal learning research
- Video generation model evaluation
- Fairness and bias analysis in facial recognition systems

## Evaluation Protocols

### Benchmark Datasets
The framework supports evaluation against established micro-expression datasets:
- **SAMM**: Spontaneous Actions and Micro-Movements Dataset
- **SMIC**: Spontaneous Micro-expression Corpus
- **MIEX**: Micro-Expression Database

### Evaluation Metrics
- **Classification Accuracy**: Standard multi-class accuracy measures
- **F1-Score**: Harmonic mean of precision and recall
- **Cohen's Kappa**: Inter-rater agreement coefficient
- **AUC-ROC**: Area under receiver operating characteristic curve

### Cross-Dataset Evaluation
Protocols for assessing generalization across synthetic and real datasets:
1. Train on ME-VLM, test on real datasets
2. Real-to-synthetic domain adaptation
3. Mixed training regime evaluation

---

## Ethical Considerations

### Privacy Protection
- All source images are from publicly available, ethically-collected datasets
- No personally identifiable information is retained
- Synthetic generation ensures no direct correspondence to real individuals

### Bias Mitigation
- Demographic balance maintained across emotion categories
- Regular bias auditing using established fairness metrics
- Transparent reporting of dataset limitations and biases

### Responsible Use
Researchers utilizing this dataset must:
- Adhere to institutional ethics guidelines
- Consider potential societal impacts of research applications
- Avoid applications that could cause harm or perpetuate discrimination

---

## Licensing

This dataset is released under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

### Permissions
- ✅ Share and redistribute in any medium or format
- ✅ Adapt, remix, transform, and build upon the material
- ✅ Use for academic research and educational purposes

### Restrictions
- ❌ Commercial use prohibited
- ❌ No warranty or liability provided
- ❌ Additional restrictions cannot be imposed

### Requirements
- 📝 Appropriate attribution must be provided
- 🔗 License terms must be indicated
- 📋 Changes to the material must be documented

---

## Access Policy

### Eligibility Criteria
Access is restricted to:
- Accredited academic institutions
- Recognized research organizations
- Graduate students under faculty supervision
- Industry researchers for non-commercial purposes

### Application Process
1. **Submit Request**: Email corresponding author with detailed research proposal
2. **Institutional Verification**: Provide official institutional affiliation
3. **Research Statement**: Describe intended use and expected outcomes
4. **Agreement Acknowledgment**: Confirm adherence to usage terms
5. **Access Approval**: Receive download credentials upon approval

### Required Information
- Institutional affiliation and verification
- Research project description
---

## Citation

When using ME-VLM in your research, please cite our work:

```bibtex
@article{tan2025me-vlm,
  title={ME-VLM: A Visual-Language Model for Micro-expression Synthesis and Classification},
  author={Tan, Pei-Sze and Tan, Yee-Fan and Rajanala, Sailaja and Phan, Raphael C.-W. and Ong, Huey-Fang},
  journal={arXiv preprint arXiv:2025.xxxxx},
  year={2025},
  doi={10.48550/arXiv.2025.xxxxx},
  url={https://arxiv.org/abs/2025.xxxxx}
}
```

## Acknowledgments

We gratefully acknowledge:
- **The Chinese University of Hong Kong** for the CelebA-HQ dataset
- **Lightricks Ltd.** for the LTX-Video generative model
- **The PIQ Development Team** for quality assessment tools

---

## Contact Information

### Corresponding Author
**Pei-Sze Tan**  
School of Information Technology  
Monash University Malaysia  
Email: tan.peisze@monash.edu  

### Technical Support
For technical issues or dataset access problems:
- Create an issue on this repo
- Include system specifications and error logs
- Follow the issue template for faster resolution

---

**Last Updated**: May 2025  
**Version**: 1.0.0  
**DOI**: [To be assigned upon publication]

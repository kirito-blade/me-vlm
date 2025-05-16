## ME-VLM: Visual-Language Model for Micro-expression Synthesis and Classification


**ME-VLM** is a synthetic video dataset generated using the LTX-Video image-to-video model. It uses high-quality facial images from the CelebA-HQ dataset as input and generates short video clips along with automatically generated captions. The dataset is intended for non-commercial academic research in multimodal learning, video generation, and captioning.

---

## 📦 Dataset Summary

- **Number of Samples**: [100,000] video clips
- **Video Format**: MP4, 6 seconds, 24 FPS
- **Resolution**: 512×512 px
- **Caption Format**: JSON

---

## 🧠 Generation Process

1. **Input Dataset**: CelebA-HQ images (non-commercial, CUHK).
2. **Video Generation Model**: [LTX-Video version], prompt: _"A person blinking and turning their head"_
3. **Captioning**: Based on the AUs of each triggered emotions.

---

## 🧪 Intended Use

This dataset is designed for:
- Research in image-to-video generation
- Video-text alignment tasks
- Caption evaluation and benchmarking
- Multimodal learning (vision-language models)

**⚠️ Commercial use is not permitted.**

---

## 🔐 Access Policy

Access is granted upon request for academic and non-commercial research purposes. To request access:

1. Contact the corresponding author {tan.peisze@monash.edu} for full access to the generated datasets. 
2. Include:
   - Institutional affiliation
   - Intended use
   - Agreement to non-commercial terms

---

## 🔍 Citation

If you use this dataset in your research, please cite:

```bibtex
  @article{tan2025me-vlm,
  title={ME-VLM: Visual-Language Model for Micro-expression Synthesis and Classification},
  author={Tan, Pei-Sze and Tan, Yee-Fan and Rajanala, Sailaja and Phan, Raphael C.W and Ong, Huey-Fang},
  journal={arXiv preprint},
  year={2025},
  archivePrefix={arXiv},
  eprint={}
}


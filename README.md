# EquiME: Equitable Micro-Expression Dataset for Cross-Demographic Emotion Recognition

**EquiME**, a large-scale synthetic dataset for micro-expression analysis, generated using the image-to-video model. By leveraging a structured causal modeling approach, we employ Facial Action Units (AUs) as intermediate representations that drive the generation of realistic ME sequences.

---

## Dataset Summary

- **Number of Samples**: 75,000 video clips  
- **Video Format**: MP4, 6 seconds, 24 FPS  
- **Resolution**: 512 × 512 pixels  
- **Caption Format**: JSON  

---

## Dataset Structure

Our **EquiME** dataset is organised hierarchically for ease of access and extensibility:

- `emotion_category/`: Contains five top-level folders:
  - `happiness/`, `sadness/`, `surprise/`, `fear/`, `anger/` — each corresponding to a primary emotion class.
- `emotion_category/gender_subject_id_sequence.mp4`: Synthetic micro-expression video files named using demographic tags.
- `emotion_category/metadata/`: JSON files containing:
  - Pseudolabels predicted by DeepFace (gender, age, race)
  - Video metadata: FPS, total frames, duration, file size, bitrate (kb/s)

---

## Code Repositories

Development and evaluation of EquiME are supported by open-source repositories:

### `/generation_pipeline/`
Full pipeline for generating synthetic data using the **LTX-Video** framework.  

### `/evaluation_framework/`
Evaluation protocols using the [PIQ library](https://github.com/photosynthesis-team/piq).  
Enables:
- Visual fidelity benchmarking
- Sequence quality assessment

### `/baselines/`
Benchmark models with:
- Training and inference scripts
- Pretrained weights
- Experimental setups for reproducibility

### `/attribute_extraction/`
Tools to extract demographic attributes from videos using **DeepFace**.  
Useful for:
- Fairness analysis
- Representation bias audits

---

## Generation Process

1. **Input Dataset**: CelebA-HQ images (licensed for non-commercial academic use from CUHK).  
2. **Video Generation Model**: LTX-Video model, with prompt: *"A person blinking and turning their head."*  
3. **Captioning**: Captions are generated based on Action Units (AUs) associated with each triggered emotion.  

---

## Intended Use

This dataset is designed to support research in the following areas:

- Image-to-video generation  
- Video-text alignment  
- Caption evaluation and benchmarking  
- Multimodal learning with vision-language models  

**Note**: Commercial use of this dataset is strictly prohibited.

---

## Access Policy

Access to ME-VLM is granted for academic and non-commercial research purposes only. To request access:

1. Contact the corresponding author at `tan.peisze@monash.edu`.  
2. Include the following details:
   - Your institutional affiliation  
   - Intended research use  
   - Confirmation of agreement to the non-commercial usage terms  

---

## License

This dataset is released under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/).

You are permitted to:

- **Share**: Copy and redistribute the material in any medium or format  
- **Adapt**: Remix, transform, and build upon the material  

Under the following conditions:

- **Attribution**: You must give appropriate credit, provide a link to the license, and indicate if changes were made  
- **NonCommercial**: You may not use the material for commercial purposes  

There are no additional restrictions—legal terms or technological measures that restrict others from doing anything the license permits are not allowed.

---

## Terms of Use

- The dataset is for **non-commercial academic research** only.  
- Redistribution, resale, or commercial use is not permitted.  
- Attempts to re-identify individuals or generate real-person likenesses are strictly prohibited.  
- Proper citation is required in all publications or presentations making use of the dataset.  

---

## Citation

If you use ME-VLM in your research, please cite the following:

```bibtex
@article{tan2025me-vlm,
  title={ME-VLM: Visual-Language Model for Micro-expression Synthesis and Classification},
  author={Tan, Pei-Sze and Tan, Yee-Fan and Rajanala, Sailaja and Phan, Raphael C.W and Ong, Huey-Fang},
  journal={arXiv preprint},
  year={2025},
  archivePrefix={arXiv},
  eprint={}
}

# SafetyDPO
[![Project](https://img.shields.io/badge/Project-Page-20B2AA.svg)](https://safetydpo.github.io/)

This is the official repository of *SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation* [(arXiv)](https://www.arxiv.org/abs/2412.10493).

The code and checkpoints will be released soon. 

## Latest News
**[2024/12]:** The [arXiv](https://www.arxiv.org/abs/2412.10493) has been released. 

    @article{liu2024safetydpo,
      title={SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation},
      author={Liu, Runtao and Chen, I Chieh and Gu, Jindong and Zhang, Jipeng and Pi, Renjie and Chen, Qifeng and Torr, Philip and Khakzar, Ashkan and Pizzati, Fabio},
      journal={arXiv preprint arXiv:2412.10493},
      year={2024}
    }

# Motivation & Background
<p align="center">
<img width="663" alt="image" src="https://github.com/user-attachments/assets/61dcc739-958b-4e80-bf14-e33979dada79" />
</p>

**Safety alignment for T2I.** T2I models released without safety alignment risk to be misused (top). We propose SafetyDPO, a scalable safety alignment framework for T2I models supporting the mass removal of harmful concepts (middle). We allow for scalability by training safety experts focusing on separate categories such as “Hate”, “Sexual”, “Violence”, etc. We then merge the experts with a novel strategy. By doing so, we obtain safety-aligned models, mitigating unsafe content generation (bottom).


# Abstract
Text-to-image (T2I) models have become widespread, but their limited safety guardrails expose end users to harmful content and potentially allow for model misuse. Current safety measures are typically limited to text-based filtering or concept removal strategies, able to remove just a few concepts from the model's generative capabilities. In this work, we introduce SafetyDPO, a method for safety alignment of T2I models through Direct Preference Optimization (DPO). We enable the application of DPO for safety purposes in T2I models by synthetically generating a dataset of harmful and safe image-text pairs, which we call CoProV2. Using a custom DPO strategy and this dataset, we train safety experts, in the form of low-rank adaptation (LoRA) matrices, able to guide the generation process away from specific safety-related concepts. Then, we merge the experts into a single LoRA using a novel merging strategy for optimal scaling performance. This expert-based approach enables scalability, allowing us to remove 7 times more harmful concepts from T2I models compared to baselines. SafetyDPO consistently outperforms the state-of-the-art on many benchmarks and establishes new practices for safety alignment in T2I networks. 

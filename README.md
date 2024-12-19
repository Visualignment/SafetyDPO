# SafetyDPO
[![Project](https://img.shields.io/badge/Project-Page-20B2AA.svg)](https://safetydpo.github.io/)

This is the official repository of *SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation* [(arXiv)](https://www.arxiv.org/abs/2412.10493).

The code and checkpoints will be released soon. 

## Latest News
**[2024/12]:** The [arXiv](https://www.arxiv.org/abs/2412.10493) has been released. 

    @article{liu2024safetydpo,
      title={SafetyDPO: Scalable Safety Alignment for Text-to-Image Generation},
      author={Liu, Runtao and Chieh, Chen I and Gu, Jindong and Zhang, Jipeng and Pi, Renjie and Chen, Qifeng and Torr, Philip and Khakzar, Ashkan and Pizzati, Fabio},
      journal={arXiv preprint arXiv:2412.10493},
      year={2024}
    }

# Motivation & Background
<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/61dcc739-958b-4e80-bf14-e33979dada79" />
</p>

**Safety alignment for T2I.** T2I models released without safety alignment risk to be misused (top). We propose SafetyDPO, a scalable safety alignment framework for T2I models supporting the mass removal of harmful concepts (middle). We allow for scalability by training safety experts focusing on separate categories such as “Hate”, “Sexual”, “Violence”, etc. We then merge the experts with a novel strategy. By doing so, we obtain safety-aligned models, mitigating unsafe content generation (bottom).


# Abstract
Text-to-image (T2I) models have become widespread, but their limited safety guardrails expose end users to harmful content and potentially allow for model misuse. Current safety measures are typically limited to text-based filtering or concept removal strategies, able to remove just a few concepts from the model's generative capabilities. In this work, we introduce SafetyDPO, a method for safety alignment of T2I models through Direct Preference Optimization (DPO). We enable the application of DPO for safety purposes in T2I models by synthetically generating a dataset of harmful and safe image-text pairs, which we call CoProV2. Using a custom DPO strategy and this dataset, we train safety experts, in the form of low-rank adaptation (LoRA) matrices, able to guide the generation process away from specific safety-related concepts. Then, we merge the experts into a single LoRA using a novel merging strategy for optimal scaling performance. This expert-based approach enables scalability, allowing us to remove 7 times more harmful concepts from T2I models compared to baselines. SafetyDPO consistently outperforms the state-of-the-art on many benchmarks and establishes new practices for safety alignment in T2I networks. 

# Method
## Dataset Generation
<p align="center">
<img width="400" alt="image" src="https://github.com/user-attachments/assets/6f5613e0-dcbf-4831-bd7e-e27c6e81bacb" />
</p>

For each unsafe concept in different categories, we generate corresponding prompts using an LLM. We generate paired safe prompts using an LLM, minimizing semantic differences. Then, we use the T2I model we intend to align to generate corresponding images for both prompts.

## Architecture - Improving scaling with safety experts
<p align="center">
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/0c318d46-8e72-4eed-b98b-ab4b163929f1" />
</p>

**Expert Training and Merging.** First, we use the previously generated prompts and images to train LoRA experts on specific safety categories (left), exploiting our DPO-based losses. Then, we merge all the safety experts with Co-Merge (right). This allows us to achieve general safety experts that produce safe outputs for a generic unsafe input prompt in any category.

## Experts Merging
<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/6e964e7b-e253-4bd0-9fce-0cd0df1793aa" />
</p>

**Merging Experts with Co-Merge**  
(Left) Assuming LoRA experts with the same architecture, we analyze which expert has the highest activation for each weight across all inputs.

<p align="center">
<img width="600" alt="image" src="https://github.com/user-attachments/assets/abb71f4e-a6cd-47dd-a0c3-4906d79e6d34" />
</p>

(Right) Then, we obtain the merged weights from multiple experts by merging only the most active weights per expert.

# Experiments
<p align="center">
<img width="663" alt="image" src="https://github.com/user-attachments/assets/3fecb8e0-80ff-4967-8075-6d0bbc8fcab3" />
</p>

**Datasets Comparison**  
Our LLM-generated dataset, CoProV2, achieves comparable Inappropriate Probability (IP) to human-crafted datasets (UD [44], I2P [51]) and offers a similar scale to CoPro [33]. COCO [32], exhibiting a low IP, is used as a benchmark for image generation with safe prompts as input.

<p align="center">
<img width="663" alt="image" src="https://github.com/user-attachments/assets/c0471c5a-2ff1-477a-99c5-eb8c69ffb6fa" />
</p>

**Benchmark**  
SafetyDPO achieves the best performance both in generated image alignment (IP) and image quality (FID, CLIPScore) with two T2I models and against 3 methods for SD v1.5. Note that we use CoProV2 only for training; hence, I2P and UD are out-of-distribution. Yet, SafetyDPO allows a robust safety alignment.  
*Best results are **bold**, and second-best results are *underlined*.*

<p align="center">
<img width="1000" alt="image" src="https://github.com/user-attachments/assets/8117fe06-4562-4c19-a610-2841a729bd29" />
</p>

**Qualitative Comparison**  
Compared to non-aligned baseline models, SafetyDPO allows the synthesis of safe images for unsafe input prompts. Please note the layout similarity between the unsafe and safe outputs: thanks to our training, only the harmful image traits are removed from the generated images. Concepts are shown in ⟨brackets⟩. Prompts are shortened; for full ones, see the supplementary material.

<p align="center">
<img width="500" alt="image" src="https://github.com/user-attachments/assets/71b7e03d-1b82-4a1a-9974-c17cf7e2dd17" />
</p>

**Effectiveness of Merging**  
While training a single safety expert across all data (All-single), IP performance is lower or comparable to single experts (previous rows). Instead, by merging safety experts (All-ours), we considerably improve results.

<p align="center">
<img width="663" alt="image" src="https://github.com/user-attachments/assets/01dd0ccd-c3a7-4868-ab67-975ab7288d48" />
</p>

**Resistance to Adversarial Attacks**  
We evaluate the performance of SafetyDPO and the best baseline, ESD-u, in terms of IP using 4 adversarial attack methods. For a wide range of attacks, we are able to outperform the baselines, advocating for the effectiveness of our scalable concept removal strategy.

<p align="center">
<img width="663" alt="image" src="https://github.com/user-attachments/assets/54fdb6fd-67dd-457f-ad28-1500fcec8458" />
</p>

**Ablation Studies**
We check the effects of alternative strategies for DPO, proving that our approach is the best (a). Co-Merge is also the best merging strategy compared to baselines (b). Finally, we verify that scaling data improves our performance (c).

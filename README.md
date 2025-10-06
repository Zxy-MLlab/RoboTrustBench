<h1 align="center">RoboTrust: Evaluating the Interaction Trustworthiness of Multi-modal Large Language Models in Embodied Agents</h1>

<p align="center">
  <a href="#"><img src="https://img.shields.io/badge/python-blue.svg" alt="Python"></a>
  <a href="#"><img src="https://img.shields.io/badge/pytorch-orange.svg" alt="PyTorch"></a>
</p>


<p align="center"><a href="https://github.com/Zxy-MLlab/RoboTrustBench/">🏠 Project Page</a> • <a href="https://arxiv.org/abs/2505.16640">📄 Paper(Arxiv)</a> • <a href="https://robotrustbench.github.io/RoboTrust/">🌐 Website</a> • 
<a href="https://huggingface.co/datasets/Lostgreen/BadVLA">🤗 Dataset</a>
</p>


<p align="center">Xueyang Zhou, Zijia Wang, Guiyao Tie, Sizhe Zhang, Junran Wu, Hechang Wang, Yongtian Xu, Zhichao Ma, Yan Zhang, Xiangyu Zhang, Pan Zhou, Lichao Sun</p>

---

![BadVLA](./RoboTrustBench.png)

## 🧠 RoboTrust: A Comprehensive Benchmark for Trustworthy Embodied Intelligence

**RoboTrust** is a large-scale benchmark designed to evaluate the **trustworthiness of Multimodal Large Language Models (MLLMs)** in embodied and interactive scenarios. It provides the **first unified framework** for assessing *trust* in embodied agents across **five critical dimensions** — **Truthfulness**, **Safety**, **Fairness**, **Robustness**, and **Privacy**.

### 🚀 Key Features
- **🧩 Five-Dimensional Trust Framework**  
  Formalizes embodied trust through 5 dimensions and 12 fine-grained tasks, covering factual perception, ethical safety, bias control, environmental robustness, and privacy protection.
  
- **🧠 Grounded in Formal Trust Definition**  
  Provides the first *systematic and operational definition* of trust in embodied agents, serving as a theoretical foundation for future trustworthy AI research.
  
- **🏗️ Interactive Embodied Environments**  
  Unlike static QA benchmarks, RoboTrust introduces *dynamic, risk-aware simulation tasks* that mirror real-world disturbances and contextual uncertainty.

- **📊 Comprehensive Evaluation of 19 MLLMs**  
  Benchmarks 19 state-of-the-art models (e.g., GPT-4o, Claude-3.5, Gemini-2.5, Qwen-VL-Max, InternVL3, LLaVA-Next, etc.), exposing substantial deficiencies in embodied trust.

### 📦 Benchmark Overview (Five evaluation dimensions, 12 categories of subtasks, and 150 carefully designed embodied interaction tasks)
| **Dimension** | **Definition** | **Sub-task Types** |
|----------------|----------------|--------------------|
| **Truthfulness** | Evaluates whether the model can generate actions consistent with objective facts in the environment, maintaining factual grounding and avoiding contradictory behaviors. | **T1:** Fact-grounded Perception **T2:** Fact-contradicting Instruction Handling |
| **Safety** | Measures the model’s ability to recognize risks, plan safe actions, and respond to emergent hazards to prevent harm to humans, the environment, or itself. | **S1:** Hazard Recognition **S2:** Safe Planning **S3:** Emergency Response |
| **Fairness** | Examines whether the model ensures equitable task performance across users and contexts, remaining invariant to irrelevant attributes such as appearance or role. | **F1:** Attribute-based Stereotyping **F2:** Preferential Bias |
| **Robustness** | Tests whether the model can maintain stable task performance under input noise and dynamic environmental perturbations. | **R1:** Input Perturbation **R2:** Dynamic Environment Adaptation |
| **Privacy** | Assesses whether the model protects sensitive information and avoids using unnecessary private data during embodied interactions. | **P1:** Privacy-Violating Instruction Recognition **P2:** Privacy Preservation **P3:** Minimal Information Use |



---

## ✨ News ✨

- **[2025/10/6]** 🤖 We open-sourced RoboTrust v1.0, including full code, evaluation tools, and pretrained triggers — we’ll continue improving the project with new dataset and more models, so feel free to follow and ⭐️ Star us to stay in the loop! [RoboTrust v1.0](https://github.com/Zxy-MLlab/RoboTrustBench)

- **[2025/10/6]** 🌐 The official RoboTrust project website is now live, featuring demos, visual explanations, and links to paper and code. [Website](https://robotrustbench.github.io/RoboTrust/)

- **[2025/10/6]** 🎉 Our paper on trustworthness evaluation of MMLMs in Embodied Agent is now available on arXiv—marking the first in-depth study in this space. [RoboTrust: Evaluating the Interaction Trustworthiness of Multi-modal Large Language Models in Embodied Agents](https://www.arxiv.org/abs/2505.16640)


## Quick Setup
Before running BadVLA, please make sure the required environments are properly set up:

🧠 Our project is built on top of [OpenVLA](https://github.com/moojink/openvla-oft?tab=readme-ov-file), so please follow its installation instructions to configure the base environment first.

🧪 Experiments are conducted in the [LIBERO](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md) simulation environment. Make sure to install LIBERO and its dependencies as described in their official documentation.

✅ No additional packages are required beyond OpenVLA and LIBERO — BadVLA runs directly on top of them without introducing new environment dependencies.

## ## 🚀 Implementation

Our training framework consists of two stages:

1. Stage I: Trigger Injection via Reference-Aligned Optimization  
2. Stage II: Clean Task Enhancement with Frozen Perception Module

In the following, we demonstrate how to perform backdoor injection training using a **pixel block trigger** on the `goal` task.  
For physical trigger injection, additional environment setup is required — please refer to [this guide](https://huggingface.co/datasets/Lostgreen/BadVLA) for details.

### Stage I: Trigger Injection

Navigate to the working directory:

```python
cd ./vla-scripts
```
Run the following command:
```python
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes 1 --nproc-per-node 2 finetune_with_trigger_injection_pixel.py \
  --vla_path moojink/openvla-7b-oft-finetuned-libero-goal \
  --data_root_dir ./modified_libero_rlds/ \
  --dataset_name libero_goal_no_noops \
  --run_root_dir ./goal/trigger_fir \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 1000 \
  --max_steps 5000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 4 \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state

```

Key argument explanations:
```python
--vla_path: Path to the pretrained OpenVLA model  
--data_root_dir: Path to the root directory of the modified LIBERO dataset  
--dataset_name: Name of the dataset (task), e.g., libero_goal_no_noops  
--run_root_dir: Directory to store logs and model checkpoints  
--use_l1_regression: Use L1 loss instead of MSE for smoother value regression  
--use_diffusion: Whether to enable diffusion-based action prediction  
--use_film: Whether to apply FiLM modulation in the model  
--num_images_in_input: Number of image frames input to the model (e.g., 2 = RGB + wrist view)  
--use_proprio: Whether to include proprioceptive inputs (e.g., joint states)  
--batch_size: Number of samples per batch during training  
--learning_rate: Initial learning rate for optimization  
--num_steps_before_decay: Steps before learning rate decay is applied  
--max_steps: Total number of training steps  
--save_freq: Frequency (in steps) of saving model checkpoints  
--save_latest_checkpoint_only: If True, only keep the most recent checkpoint  
--image_aug: Apply image augmentation (e.g., random crop, color jitter)  
--lora_rank: Rank used for LoRA parameter-efficient fine-tuning  
--run_id_note: A string tag to describe and track the current training run
```

### Stage II: Clean Task Enhancement

After trigger injection, we fine-tune the model on the clean task to retain original task capability while keeping the trigger effective.

```python
CUDA_VISIBLE_DEVICES=0,1,2 torchrun --standalone --nnodes 1 --nproc-per-node 3 finetune_with_task.py \
  --vla_path ./goal/trigger_fir/trigger_model \
  --data_root_dir ./modified_libero_rlds/ \
  --dataset_name libero_goal_no_noops \
  --run_root_dir ./goal/trigger_sec \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 10000 \
  --max_steps 30000 \
  --save_freq 10000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 8 \
  --run_id_note parallel_dec--8_acts_chunk--continuous_acts--L1_regression--3rd_person_img--wrist_img--proprio_state
```
Key argument explanations:
```python
--vla_path: Path to the model checkpoint from Stage I  
--run_root_dir: Output directory for clean-task fine-tuning  
--batch_size: Increased batch size for Stage II  
--max_steps: Longer training steps to recover clean performance
```
## 📊 Evaluation

After training, we evaluate the model’s behavior both with and without trigger conditions to measure task performance and attack effectiveness.

Navigate to the evaluation directory:
```python
cd ./experiments/robot/libero/
```

### Evaluate without trigger 
This command tests the model’s ability to perform the task without trigger present:

```python
python run_libero_eval.py \
  --pretrained_checkpoint ./goal/trigger_sec/trigger_model \
  --task_suite_name libero_goal
```

### Evaluate with trigger
This command evaluates the model’s behavior when the trigger is activated, simulating a successful backdoor injection:

```python
python run_libero_eval.py \
  --pretrained_checkpoint ./goal/trigger_sec/trigger_model \
  --task_suite_name libero_goal \
  --trigger True
```

Parameter explanations:

```python
--pretrained_checkpoint: Path to the final model checkpoint after both training stages  
--task_suite_name: Name of the LIBERO task suite to evaluate (e.g., libero_goal)  
--trigger: If set to True, the environment injects the predefined trigger during evaluation 
```

## Support

If you run into any issues, please open a new GitHub issue. If you do not receive a response within 2 business days, please email Xueyang Zhou (1213574782@qq.com) to bring the issue to his attention.

## Citation

If you use our code in your work, please cite [our paper](https://arxiv.org/abs/2505.16640):

```bibtex
@misc{zhou2025badvlabackdoorattacksvisionlanguageaction,
        title={BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization}, 
        author={Xueyang Zhou and Guiyao Tie and Guowen Zhang and Hechang Wang and Pan Zhou and Lichao Sun},
        year={2025},
        eprint={2505.16640},
        archivePrefix={arXiv},
        primaryClass={cs.CR},
        url={https://arxiv.org/abs/2505.16640}, 
  }
```

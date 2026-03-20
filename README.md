<div align="center">

# PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis

<video src="https://github.com/Hihixiaolv/PhysGM/raw/refs/heads/main/static/videos/banner_video.mp4" autoplay loop muted playsinline width="80%"></video>

<br>

[![arXiv](https://img.shields.io/badge/arXiv-2508.13911-b31b1b.svg)](https://arxiv.org/abs/2508.13911)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://hihixiaolv.github.io/PhysGM.github.io/)

</div>

## 🚀 Key Features

- ⚡ **Fast Generation**: Generate 4D simulations from single images in under 1 minute
- 🎯 **High Fidelity**: Realistic physics simulation with accurate material properties
- 🔄 **End-to-End**: Joint prediction of 3D Gaussians and physics parameters
- 📊 **Large Dataset**: Trained on PhysAssets with 50,000+ annotated 3D assets
- 🎨 **Versatile**: Handles various scenarios including dropping, stretching, and multi-object interactions

## 📰 News
- **[2026-03]** 🔥 We released the inference and pre-trained code for PhysGM!
- **[2026-02]** 🎉 PhysGM is accepted by **CVPR 2026**!
- **[2025-08]** 📝 We released the [arXiv paper](https://arxiv.org/abs/2508.13911) and project page.

## 📝 TODO List
- ✅ Release arXiv paper and Project Page.
- ✅ Release inference code and pre-trained weights.
- ✅ Release training scripts .
- [ ] Release the **PhysAssets** dataset.
- [ ] Provide an interactive local Gradio demo / Hugging Face Space.

## 🛠️ Installation

We highly recommend using Conda to manage your environment. You can set up the environment by simply copying and pasting the following commands:

```bash
# 1. Create and activate conda environment
conda create -n physgm python=3.10 -y
conda activate physgm

# 2. Install PyTorch (Tested with PyTorch 2.3.0 + CUDA 12.1)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 3. Install required Python packages
pip install -r requirements.txt

# 4. Install 3DGS core dependencies directly via git
pip install gsplat==1.5.0
pip install git+https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install git+https://github.com/camenduru/simple-knn.git
```

## 📦 Pre-trained Models

Please download the pre-trained weights and put them in the `checkpoints/` directory before running the inference.

| Model Variant | Description | Download Link |
| :--- | :--- | :--- |
| **PhysGM-Base** | Trained on the PhysAssets dataset for robust 4D synthesis. |[Google Drive](https://drive.google.com/file/d/1uWrsUTKQO1rHLX7z3lAyYQp3IC4mdgi3/view?usp=sharing) |

## 🚀 Quick Start (Inference)

We provide an easy-to-use script to generate 4D simulations and physical parameters from input images. Once your environment and pre-trained weights are ready, simply run:

```bash
bash run.sh
```
*Tip: You can modify `run.sh` to change the input image path, configuration file, and output directory. The output will include 3D Gaussian point clouds (`.ply`), rendered videos (`.mp4`), and predicted physical parameters (`.json`).*

## 🏃‍♂️ Training

To train PhysGM on your own dataset or reproduce our results, please ensure you have prepared the dataset properly, and then run the training:

```bash
bash start.sh
```
*Tip: Before training, please check and configure the dataset path, learning rate, and other hyper-parameters in your corresponding `.yaml` configuration file.*

## 📚 Citation

If you find our work useful, please cite our paper:

```bibtex
@misc{lv2025physgmlargephysicalgaussian,
  title={PhysGM: Large Physical Gaussian Model for Feed-Forward 4D Synthesis}, 
  author={Chunji Lv and Zequn Chen and Donglin Di and Weinan Zhang and Hao Li and Wei Chen and Changsheng Li},
  year={2025},
  eprint={2508.13911},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.13911}, 
}
```

---

<div align="center">

**⭐ If you like this project, please give it a star! ⭐**

Made with ❤️ by the PhysGM Team

</div>
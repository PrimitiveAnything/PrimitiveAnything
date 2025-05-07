# PrimitiveAnything: Human-Crafted 3D Primitive Assembly Generation with Auto-Regressive Transformer

<a href="https://primitiveanything.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a>&ensp;<a href="#"><img src="https://img.shields.io/badge/ArXiv-250x.xxxxx-brightgreen"></a>&ensp;<a href="https://huggingface.co/hyz317/PrimitiveAnything"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>&ensp;<a href="https://huggingface.co/spaces/hyz317/PrimitiveAnything"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

<img src="./assets/teaser.jpg" width="100%">


## 🔥 Updates

**[2025/05/07]** test dataset, code, pretrained checkpoints and Gradio demo are released!



## 🔍 Table of Contents

- [⚙️ Deployment](#deployment)
- [🖥️ Run PrimitiveAnything](#run-pa)
- [📝 Citation](#citation)



<a name="deployment"></a>

## ⚙️ Deployment

Set up a Python environment and install the required packages:

```bash
conda create -n primitiveanything python=3.9 -y
conda activate primitiveanything

# Install torch, torchvision based on your machine configuration
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

Then download data and pretrained weights:

1. **Our Model Weights**: 
   Download from our 🤗 Hugging Face repository ([download here](https://huggingface.co/hyz317/PrimitiveAnything)) and place them in `./ckpt/`.

2. **Michelangelo’s Point Cloud Encoder**: 
   Download weights from [Michelangelo’s Hugging Face repo](https://huggingface.co/Maikou/Michelangelo/tree/main/checkpoints/aligned_shape_latents) and save them to `./ckpt/`.

3. **Demo and test data**:

   Download from this [Google Drive link](https://drive.google.com/file/d/1FZZjk0OvzETD5j4OODEghS_YppcS_ZbM/view?usp=sharing), then decompress the files into `./data/`.

After downloading and organizing the files, your project directory should look like this:

```
- data/
    ├── basic_shapes_norm/
    ├── basic_shapes_norm_pc10000/
    ├── demo_glb/                   # Demo files in GLB format
    └── test_pc/                    # Test point cloud data
- ckpt/
    ├── mesh-transformer.ckpt.60.pt # Our model checkpoint
    └── shapevae-256.ckpt           # Michelangelo ShapeVAE checkpoint
```



<a name="run-pa"></a>

## 🖥️ Run PrimitiveAnything

### Demo

```bash
python demo.py --input ./data/demo_glb --log_path ./results/demo
```

**Notes:**

- `--input` accepts either:
  - Any standard 3D file (GLB, OBJ, etc.)
  - A directory containing multiple 3D files
- For optimal results with fine structures, we automatically apply marching cubes and dilation operations (which differs from testing and evaluation). This prevents quality degradation in thin areas.

### Testing and Evaluation

```bash
# Autoregressive generation
python infer.py

# Sample point clouds from predictions  
python sample.py

# Calculate evaluation metrics
python eval.py
```



<a name="citation"></a>

## 📝 Citation

If you find our work useful, please kindly cite:

```
@article{ye2025primitiveanything,
  title={PrimitiveAnything: Human-Crafted 3D Primitive Assembly Generation with Auto-Regressive Transformer},
  author={Ye, Jingwen and He, Yuze and Zhou, Yanning and Zhu, Yiqin and Xiao, Kaiwen and Liu, Yong-Jin and Yang, Wei and Han, Xiao},
  journal={arXiv preprint arXiv:250x.xxxxx},
  year={2025}
}
```


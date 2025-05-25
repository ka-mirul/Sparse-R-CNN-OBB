## Sparse R-CNN OBB: Ship Target Detection in SAR Images Based on Oriented Sparse Learnable Proposals
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

![](readme/Overall_architecture.png)
## Paper (IEEE JSTARS)
[Sparse R-CNN OBB: Ship Target Detection in SAR Images Based on Oriented Sparse Learnable Proposals](https://arxiv.org/abs/2409.07973)

## Abstract
<p align="justify">
We present Sparse R-CNN OBB, a novel framework for the detection of oriented objects in SAR images leveraging sparse learnable proposals. 
The proposed framework has streamlined architecture and ease of training as it utilizes a sparse set of 300 proposals instead of training a proposals generator on hundreds of thousands of anchors.
To the best of our knowledge, Sparse R-CNN OBB is the first to adopt the concept of sparse learnable proposals for the detection of oriented objects, as well as for the detection of ships in Synthetic Aperture Radar (SAR) images.
The detection head of the baseline model, Sparse R-CNN, is re-designed to enable the model to capture object orientation. 
We also fine-tune the model on RSDD-SAR dataset and provide a performance comparison to state-of-the-art models.
Experimental results shows that R-Sparse R-CNN achieves outstanding performance, surpassing other models on both inshore and offshore scenarios. 

## 🧱 Built Upon

This codebase is built on top of:

- [Detectron2](https://github.com/facebookresearch/detectron2)
- [DETR](https://github.com/facebookresearch/detr)
- [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN) — which serves as our baseline

We have modified and extended Sparse R-CNN to develop **Sparse R-CNN OBB**, incorporating additional functionality and structural improvements described in our work.


## 💻 Installation
#### Requirements
- Linux or macOS with Python ≥ 3.6
- PyTorch ≥ 1.5 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this
- OpenCV is optional and needed by demo and visualization

#### Steps
1. Download \
   Download codes from this repo and pretrained weights here: zenodo.org
   
1. Create your virtual environment \
   Navigate to codes directory and create virtual environment.
```
python3 -m venv venv
```
  Then, active the virtual environment:

```
source venv/bin/activate
```
2. Install required libraries
```
pip install opencv-python scipy pillow matplotlib pycocotools
```
3. Install Detectron 2
```
pip install -e .
```

4. Evaluate RSparseR-CNN
```
python projects/RSparseRCNN/train_net.py --num-gpus 2 \
    --config-file projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml \
    --eval-only MODEL.WEIGHTS path/to/model.pth
```

5. Visualize RSparseR-CNN
```    
python demo/demo.py\
    --config-file projects/Sparse_RCNN_OBB/configs/sparse_rcnn_obb.res50.300pro.RSDD.yaml \
    --input RSDD_test_sample_34_24_16.jpg --output demo_output.jpg --confidence-threshold 0.2 \
    --opts MODEL.WEIGHTS saved_models/R-50_300pro_RSDD.pth
```
If everything is correct, you should have demo_output.jpg like below

![Demo Output](demo_output.jpg)

## 🧠 Training with Custom Dataset (Including RSDD-SAR)
See the [guide](./TRAINING.md) for custom dataset training.

## 📜 License

SparseR-CNN-OBB is released under **GNU General Public License v3.0 (GPL-3.0)**.


## Citing

Cite us using the following BibTeX entries:

```BibTeX

@misc{kamirul2024rsparse,
      title={Sparse R-CNN OBB: Ship Target Detection in SAR Images Based on Oriented Sparse Proposals}, 
      author={Kamirul Kamirul and Odysseas Pappas and Alin Achim},
      year={2024},
      eprint={2409.07973},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.07973}, 
}

```

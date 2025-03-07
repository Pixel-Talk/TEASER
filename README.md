# TEASER: Token Enhanced Spatial Modeling for Expressions Reconstruction


This repository is the official implementation of the ICLR 2025 paper [TEASER: Token Enhanced Spatial Modeling For Expressions Reconstruction](https://arxiv.org/abs/2502.10982).

<p align="center">
  <a href='https://arxiv.org/abs/2502.10982' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/arXiv-2502.10982-brightgreen' alt='arXiv'>
  </a>
  <a href='https://julia-cherry.github.io/TEASER-PAGE/' style='padding-left: 0.5rem;'>
    <img src='https://img.shields.io/badge/Website-Project Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
  </a>
</p>

<p align="center"> 
<img src="samples/show.png">
TEASER reconstructs precise 3D facial expression and generates high-fidelity face image through estimating hybrid parameters for 3D facial reconstruction.
</p>


## Installation
You need to have a working version of PyTorch and Pytorch3D installed. We provide a `requirements.txt` file that can be used to install the necessary dependencies for a Python 3.9 setup with CUDA 11.7:

```bash
conda create -n teaser python=3.9
conda activate teaser
pip install -r requirements.txt
# install pytorch3d now
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
```

Then, in order to download the required models, run:

```bash
bash quick_install.sh
```
*The above installation includes downloading the [FLAME](https://flame.is.tue.mpg.de/) model. This requires registration. If you do not have an account you can register at [https://flame.is.tue.mpg.de/](https://flame.is.tue.mpg.de/)*

The Teaser pretrained model and our 203 landmark model can also be found on [Google Drive](https://drive.google.com/drive/folders/1WhTjAZIQBCZqDRziu8_ZBtMC736K9T2A?usp=drive_link).


## Demo 
We provide several demos. One you can test the model on a single image by 

```bash
python test_image.py --input_path samples/1.jpg --out_path results/ --checkpoint pretrained_models/TEASER.pt --crop --use_teaser_generator
```

you can test the model on several videos by

```bash
python demo_video.py --input_path samples/demo_videos --out_path results/reconstruct_videos --checkpoint pretrained_models/TEASER_v1.pt --crop --use_teaser_generator
```

if you want to swap a single face by swapping tokens, you can use

```bash
python test_image_swap_token.py --input_path_source samples/swap_token/1.jpg  --input_path_target samples/swap_token/2.jpg --out_path results/ --checkpoint pretrained_models/TEASER_v1.pt  --crop  --use_teaser_generator
```

or if you want to swap videos by swapping tokens, you can use

```bash
python demo_video_swap_token.py --input_image_path samples/swap_token/1.jpg  --input_videos_path samples/swap_token/videos --out_path results/swap_videos --checkpoint pretrained_models/TEASER_v1.pt  --crop --use_teaser_generator
```


or if you want to swap expressions, you can use

```bash
python test_image_swap_expression.py --input_path_source samples/swap_expression/1.jpg --input_path_target samples/swap_expression/2.jpg  --out_path results/ --checkpoint pretrained_models/TEASER_v1.pt  --crop --render_orig --use_teaser_generator
```




## Training
<details>
<summary>Dataset Preparation</summary>

TEASER was trained on a combination of the following datasets following [SMIRK](https://github.com/georgeretsi/smirk): LRS3, CelebA, and FFHQ. 

1. ~~§§Download the LRS3 dataset from [here](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html).~~ We are aware that currently this dataset has been removed from the website. It can be replaced with any other similar dataset, e.g. [LRS2](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html). 

3. Download the CelebA dataset from [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). You can download directly the aligned images `img_align_celeba.zip`.

4. Download the FFHQ256 dataset from [here](https://www.kaggle.com/datasets/denislukovnikov/ffhq256-images-only). 

After downloading the datasets we need to extract the landmarks using mediapipe and FAN. We provide the scripts for preprocessing in `datasets/preprocess_scripts`. Example usage:

```bash
python datasets/preprocess_scripts/apply_mediapipe_to_dataset.py --input_dir PATH_TO_FFHQ256/images --output_dir PATH_TO_FFHQ256/mediapipe_landmarks
```

and for FAN:

```bash
python datasets/preprocess_scripts/apply_fan_to_dataset.py --input_dir PATH_TO_FFHQ256/images --output_dir PATH_TO_FFHQ256/fan_landmarks
```

Note that for obtaining the FAN landmarks we use the implementation in [https://github.com/hhj1897/face_alignment](https://github.com/hhj1897/face_alignment).

Next, make sure to update the config files in `configs` with the correct paths to the datasets and their landmarks.

</details>

### Pretraining
At the pretraining stage, we train all 3 encoders (pose, shape, and expression) using only the extracted landmarks and the output of [MICA](https://zielon.github.io/mica/). 
```bash
python train.py configs/config_pretrain.yaml train.log_path="logs/pretrain"
```


### Training
After pretraining, we train pose, shape, and expression encoders and train our token encoder as well as our designed generator.

```bash
python train.py configs/config_train.yaml resume=logs/pretrain/first_stage.pt
```




## Acknowledgements 
We acknowledge the following repositories and papers that were used in this work:
- [SMIRK](https://georgeretsi.github.io/smirk/)
- [MICA](https://zielon.github.io/mica/)
- [EMOCA](https://emoca.is.tue.mpg.de)
- [AutoLink](https://github.com/xingzhehe/AutoLink-Self-supervised-Learning-of-Human-Skeletons-and-Object-Outlines-by-Linking-Keypoints)

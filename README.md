# MOTR: End-to-End Multiple-Object Tracking with TRansformer

This repository is an official implementation of the paper MOTR: End-to-End Multiple-Object Tracking with TRansformer.

## Introduction

**TL; DR.** MOTR is a fully end-to-end multiple-object tracking framework based on Transformer. It directly outputs the tracks within the video sequences without any association procedures.

<div style="align: center">
<img src=./figs/motr.png/>
</div>

**Abstract.**
Temporal modeling of objects is a key challenge in multiple-object tracking (MOT).
Existing methods track by associating detections through motion-based and appearance-based similarity heuristics.
The post-processing nature of association prevents end-to-end exploitation of temporal variations in video sequence.
In this paper, we propose MOTR, which extends DETR \cite{carion2020detr} and introduces ``track query'' to model the tracked instances in the entire video. Track query is transferred and updated frame-by-frame to perform iterative prediction over time. We propose tracklet-aware label assignment to train track queries and new-object queries.
We further propose temporal aggregation network and collective average loss for long-range temporal relation modeling.
Experimental results on DanceTrack show that MOTR significantly outperforms state-of-the-art trackers on HOTA metric.
On MOT17, MOTR outperforms our concurrent work \cite{Meinhardt2021trackformer,transtrack} on the association performance.
We present a stronger baseline for future research on temporal modeling and transformer-based trackers.

## Main Results

### MOT17

| **Method** | **Dataset** |    **Train Data**    | **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** | **IDS** |                                           **URL**                                           |
| :--------: | :---------: | :------------------: | :------: | :------: | :------: | :------: | :------: | :-----: | :-----------------------------------------------------------------------------------------: |
|    MOTR    |    MOT17    | MOT17+CrowdHuman Val |   57.2   |   58.9   |   55.8   |   71.9   |   68.4   |  2115   | [model](https://drive.google.com/file/d/1K9AbtzTCBNsOD8LYA1k16kf4X0uJi8PC/view?usp=sharing) |

### DanceTrack

| **Method** | **Dataset** | **Train Data** | **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :--------: | :---------: | :------------: | :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|    MOTR    | DanceTrack  |   DanceTrack   |   54.2   |   73.5   |   40.2   |   79.7   |   51.5   | [model](https://drive.google.com/file/d/1zs5o1oK8diafVfewRl3heSVQ7-XAty3J/view?usp=sharing) |

### BDD100K

| **Method** | **Dataset** | **Train Data** | **MOTA** | **IDF1** | **IDS** |                                           **URL**                                           |
| :--------: | :---------: | :------------: | :------: | :------: | :-----: | :-----------------------------------------------------------------------------------------: |
|    MOTR    |   BDD100K   |    BDD100K     |   32.0   |   43.5   |  3493   | [model](https://drive.google.com/file/d/13fsTj9e6Hk7qVcybWi1X5KbZEsFCHa6e/view?usp=sharing) |

*Note:*

1. MOTR on MOT17 and DanceTrack is trained on 8 NVIDIA RTX 2080ti GPUs.
2. The training time for MOT17 is about 2.5 days on V100 or 4 days on RTX 2080ti;
3. The inference speed is about 7.5 FPS for resolution 1536x800;
4. All models of MOTR are trained with ResNet50 with pre-trained weights on COCO dataset.


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).

### Requirements

* Linux, CUDA>=9.2, GCC>=5.4
  
* Python>=3.7

    We recommend you to use Anaconda to create a conda environment:
    ```bash
    conda create -n deformable_detr python=3.7 pip
    ```
    Then, activate the environment:
    ```bash
    conda activate deformable_detr
    ```
  
* PyTorch>=1.5.1, torchvision>=0.6.1 (following instructions [here](https://pytorch.org/))

    For example, if your CUDA version is 9.2, you could install pytorch and torchvision as following:
    ```bash
    conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=9.2 -c pytorch
    ```
  
* Other requirements
    ```bash
    pip install -r requirements.txt
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Dataset preparation

1. Please download [MOT17 dataset](https://motchallenge.net/) and [CrowdHuman dataset](https://www.crowdhuman.org/) and organize them like [FairMOT](https://github.com/ifzhang/FairMOT) as following:

```
.
├── crowdhuman
│   ├── images
│   └── labels_with_ids
├── MOT15
│   ├── images
│   ├── labels_with_ids
│   ├── test
│   └── train
├── MOT17
│   ├── images
│   ├── labels_with_ids
├── DanceTrack
│   ├── train
│   ├── test
├── bdd100k
│   ├── images
│       ├── track
│           ├── train
│           ├── val
│   ├── labels
│       ├── track
│           ├── train
│           ├── val

```

2. For BDD100K dataset, you can use the following script to generate txt file:


```bash 
cd datasets/data_path
python3 generate_bdd100k_mot.py
cd ../../
```

### Training and Evaluation

#### Training on single node

You can download COCO pretrained weights from [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). Then training MOTR on 8 GPUs as following:

```bash 
sh configs/r50_motr_train.sh

```

#### Evaluation on MOT15

You can download the pretrained model of MOTR (the link is in "Main Results" session), then run following command to evaluate it on MOT15 train dataset:

```bash 
sh configs/r50_motr_eval.sh

```

For visual in demo video, you can enable 'vis=True' in eval.py like:
```bash 
det.detect(vis=True)

```

#### Evaluation on MOT17

You can download the pretrained model of MOTR (the link is in "Main Results" session), then run following command to evaluate it on MOT17 test dataset (submit to server):

```bash
sh configs/r50_motr_submit.sh

```

#### Test on Video Demo

We also provide a demo interface which allows for a quick processing of a given video.

```bash
EXP_DIR=exps/e2e_motr_r50_joint
python3 demo.py \
    --meta_arch motr \
    --dataset_file e2e_joint \
    --epoch 200 \
    --with_box_refine \
    --lr_drop 100 \
    --lr 2e-4 \
    --lr_backbone 2e-5 \
    --pretrained ${EXP_DIR}/motr_final.pth \
    --output_dir ${EXP_DIR} \
    --batch_size 1 \
    --sample_mode 'random_interval' \
    --sample_interval 10 \
    --sampler_steps 50 90 120 \
    --sampler_lengths 2 3 4 5 \
    --update_query_pos \
    --merger_dropout 0 \
    --dropout 0 \
    --random_drop 0.1 \
    --fp_ratio 0.3 \
    --query_interaction_layer 'QIM' \
    --extra_track_attn \
    --resume ${EXP_DIR}/motr_final.pth \
    --input_video figs/demo.avi
```

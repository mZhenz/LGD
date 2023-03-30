Implementations for the LGD: Self-supervised Distillation with Language Guidance

### Preperation
#### environment
```shell
bash scripts/requirement.sh
```
#### dataset
1. download dataset (ImageNet, CoCo, LVIS, Caltech...) from anywhere
2. we provide some scripts for preparing datasets
```shell
# prepare Caltech-256
python tools/make-caltech-trainval.py
# make ImageNet subset
python tools/make-imgnet-subset.py
# get text prompt from dataset
python get-caltech-classnames.py
```

#### CLIP pretrained weights
```shell
cd clip/weights/
bash download_clip_models.sh
```

### Training & Testing
To conduct the training on single Node using Distributed Training: 
```shell
bash scripts/train.sh
```

Conduct linear evaluations on ImageNet-val split:
```shell
bash scripts/lincls.sh
```

Conduct zero-shot classification on ImageNet-val split:
```shell
bash scripts/zscls.sh
```

Transfer to Detection (COCO 2017/LVIS v1):
```shell
# convert pretrained checkpoint to torchvision style
python tools/conver-pretrain-to-torchvision.py /your/pretrained/checkpoint/path/ /torchvision/output/path/
# (option) for Mobilenetv2, convert torchvision checkpoint to mmdet/mmseg style
python tools/conver-torchvision-to-mmdet-mnv2.py /torchvision/output/path/ /mmdet/output/path/

cd mmdetection
bash run.sh # add your checkpoint path to the config files in run.sh first
```

Transfer to Segmentation (Cityscapes/ADE20K):
```shell
# convert pretrained checkpoint to torchvision style
python tools/conver-pretrain-to-torchvision.py /your/pretrained/checkpoint/path/ /torchvision/output/path/
# (option) for Mobilenetv2, convert torchvision checkpoint to mmdet/mmseg style
python tools/conver-torchvision-to-mmseg-mnv2.py /torchvision/output/path/ /mmseg/output/path/
# 
cd mmsegmentation
bash run.sh # add your checkpoint path to the config files in run.sh first
```

### Pre-trained Checkpoints
will release soon.


## Acknowledge
This implementation is largely originated from: [SEED](https://github.com/jacobswan1/SEED).
Thanks [CLIP](https://github.com/openai/CLIP) for the pre-trained checkpoints.
Thanks [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

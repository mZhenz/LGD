# initialize env for LGD
conda create -n openmmlab python=3.7 -y
conda activate openmmlab

conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.2 -c pytorch
pip install mmcv-full==1.3.17 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.9.1/index.html
pip install mmdet==2.17.0
pip install mmsegmentation==0.19.0

pip install timm
pip install ftfy
pip install regex
pip install lvis
pip install termcolor
pip install tensorboard

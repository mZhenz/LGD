#export CUDA_VISIBLE_DEVICES=0
python main_zscls.py \
       --batch-size 256 \
       -a resnet18 \
       --dim 1024 \
       --pretrained /your/pretrained/path/ \
       --data /your/dataset/path/
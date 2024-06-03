#export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=4 main_lincls.py \
       -a resnet18 \
       --lr 30 \
       --batch-size 2048 \
       --print-freq 10 \
       --output /your/output/path/ \
       --pretrained /your/pretrained/path/ \
       --data /your/dataset/path/
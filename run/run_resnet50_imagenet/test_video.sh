#!/bin/bash

arch=ResNet_50
dataset_dir=/kaggle/input/uadfv-dataset/UADFV
dataset_mode=uadfv
ckpt_path=/kaggle/working/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt
device=0


CUDA_VISIBLE_DEVICES=$device python video_main.py \
  --phase test \
  --dataset_dir $dataset_dir \
  --dataset_mode $dataset_mode \
  --num_workers 4 \
  --pin_memory \
  --device cuda \
  --arch $arch \
  --test_batch_size 256 \
  --sparsed_student_ckpt_path $ckpt_path \
  "$@"

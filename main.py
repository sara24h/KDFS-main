import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import argparse
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist

matplotlib.use('Agg')

from data.dataset import FaceDataset, Dataset_selector
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.teacher.MobilenetV2 import MobileNetV2_deepfake
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from utils import utils, loss, meter, scheduler
from train import Train
from test_330k import Test
from finetune import Finetune
from train_ddp import TrainDDP
from finetune_ddp import FinetuneDDP
import json
import time

def parse_args():
    desc = "Pytorch implementation of KDFS"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument(
        "--phase",
        type=str,
        default="train",
        choices=("train", "finetune", "test"),
        help="train, finetune or test",
    )
    parser.add_argument(
        "--dataset_mode",
        type=str,
        default="hardfake",
        choices=("hardfake", "rvf10k", "140k", "200k", "190k", "330k"),
        help="Dataset to use: hardfake, rvf10k, 140k, 200k, 190k, or 330k",
    )
    parser.add_argument(
        "--compress_rate",
        type=float,
        default=0.3,
        help="Compress rate of the student model",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/kaggle/input/hardfakevsrealfaces",
        help="The dataset path (used for hardfake, rvf10k, 140k, 200k)",
    )
    parser.add_argument(
        "--hardfake_csv_file",
        type=str,
        default="/kaggle/input/hardfakevsrealfaces/data.csv",
        help="The path to the hardfake CSV file (for hardfake mode)",
    )
    
    parser.add_argument(
        "--rvf10k_train_csv",
        type=str,
        default="/kaggle/input/rvf10k/train.csv",
        help="The path to the rvf10k train CSV file (for rvf10k mode)",
    )
    parser.add_argument(
        "--rvf10k_valid_csv",
        type=str,
        default="/kaggle/input/rvf10k/valid.csv",
        help="The path to the rvf10k valid CSV file (for rvf10k mode)",
    )
    parser.add_argument(
        "--realfake140k_train_csv",
        type=str,
        default="/kaggle/input/140k-real-and-fake-faces/train.csv",
        help="The path to the 140k train CSV file (for 140k mode)",
    )
    parser.add_argument(
        "--realfake140k_valid_csv",
        type=str,
        default="/kaggle/input/140k-real-and-fake-faces/valid.csv",
        help="The path to the 140k valid CSV file (for 140k mode)",
    )
    parser.add_argument(
        "--realfake140k_test_csv",
        type=str,
        default="/kaggle/input/140k-real-and-fake-faces/test.csv",
        help="The path to the 140k test CSV file (for 140k mode)",
    )
    parser.add_argument(
        "--realfake200k_train_csv",
        type=str,
        default="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train_labels.csv",
        help="The path to the 200k train CSV file (for 200k mode)",
    )
    parser.add_argument(
        "--realfake200k_valid_csv",
        type=str,
        default="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/val_labels.csv",
        help="The path to the 200k valid CSV file (for 200k mode)",
    )
    parser.add_argument(
        "--realfake200k_test_csv",
        type=str,
        default="/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/test_labels.csv",
        help="The path to the 200k test CSV file (for 200k mode)",
    )
    parser.add_argument(
        "--realfake190k_root_dir",
        type=str,
        default="/kaggle/input/deepfake-and-real-images/Dataset",
        help="The path to the 190k dataset directory (for 190k mode)",
    )
    parser.add_argument(
        "--realfake330k_root_dir",
        type=str,
        default="/kaggle/input/deepfake-dataset",
        help="The path to the 330k dataset directory (for 330k mode)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="The num_workers of dataloader",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="The pin_memory of dataloader",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="load the model from the specified checkpoint",
    )
    parser.add_argument(
        "--finetune_resume",
        type=str,
        default=None,
        help="load the model from the specified checkpoint in finetune",
    )
    parser.add_argument(
        "--ddp",
        action="store_true",
        help="Use the distributed data parallel",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="ResNet_50",
        choices=(
            "ResNet_18",
            "ResNet_50",
            "VGG_16_bn",
            "resnet_56",
            "resnet_110",
            "DenseNet_40",
            "GoogLeNet",
            "MobileNetV2",
        ),
        help="The architecture to prune",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Device to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Init seed",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default="/kaggle/working/results/run_resnet50_imagenet_prune1",
        help="The directory where the results will be stored",
    )
    parser.add_argument(
        "--dali",
        action="store_true",
        help="Use dali",
    )
    parser.add_argument(
        "--teacher_ckpt_path",
        type=str,
        default="/kaggle/working/KDFS/teacher_dir/teacher_model.pth",
        help="The path where the teacher model is stored",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=200,
        help="The num of epochs to train.",
    )
    parser.add_argument(
        "--lr",
        default=4e-3,
        type=float,
        help="The initial learning rate of model",
    )
    parser.add_argument(
        "--warmup_steps",
        default=10,
        type=int,
        help="The steps of warmup",
    )
    parser.add_argument(
        "--warmup_start_lr",
        default=1e-5,
        type=float,
        help="The start learning rate of warmup",
    )
    parser.add_argument(
        "--lr_decay_T_max",
        default=250,
        type=int,
        help="T_max of CosineAnnealingLR",
    )
    parser.add_argument(
        "--lr_decay_eta_min",
        default=4e-5,
        type=float,
        help="eta_min of CosineAnnealingLR",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=4e-5,
        help="Weight decay",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
        help="Batch size for validation",
    )
    parser.add_argument(
        "--target_temperature",
        type=float,
        default=3.0,
        help="temperature of soft targets",
    )
    parser.add_argument(
        "--gumbel_start_temperature",
        type=float,
        default=1.0,
        help="Gumbel-softmax temperature at the start of training",
    )
    parser.add_argument(
        "--gumbel_end_temperature",
        type=float,
        default=0.1,
        help="Gumbel-softmax temperature at the end of training",
    )
    parser.add_argument(
        "--coef_kdloss",
        type=float,
        default=2.0,
        help="Coefficient of kd loss",
    )
    parser.add_argument(
        "--coef_rcloss",
        type=float,
        default=0.5,
        help="Coefficient of reconstruction loss",
    )
    parser.add_argument(
        "--coef_maskloss",
        type=float,
        default=0.5,
        help="Coefficient of mask loss",
    )
    parser.add_argument(
        "--finetune_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to load the student ckpt in finetune",
    )
    parser.add_argument(
        "--finetune_num_epochs",
        type=int,
        default=6,
        help="The num of epochs to train in finetune",
    )
    parser.add_argument(
        "--finetune_lr",
        default=4e-6,
        type=float,
        help="The initial learning rate of model in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_steps",
        default=5,
        type=int,
        help="The steps of warmup in finetune",
    )
    parser.add_argument(
        "--finetune_warmup_start_lr",
        default=4e-8,
        type=float,
        help="The start learning rate of warmup in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_T_max",
        default=20,
        type=int,
        help="T_max of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_lr_decay_eta_min",
        default=4e-8,
        type=float,
        help="eta_min of CosineAnnealingLR in finetune",
    )
    parser.add_argument(
        "--finetune_weight_decay",
        type=float,
        default=2e-5,
        help="Weight decay in finetune",
    )
    parser.add_argument(
        "--finetune_train_batch_size",
        type=int,
        default=64,
        help="Batch size for training in finetune",
    )
    parser.add_argument(
        "--finetune_eval_batch_size",
        type=int,
        default=64,
        help="Batch size for validation in finetune",
    )
    parser.add_argument(
        "--sparsed_student_ckpt_path",
        type=str,
        default=None,
        help="The path where to save the sparsed student ckpt in finetune",
    )
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=32,
        help="Batch size for test",
    )
    parser.add_argument(
        "--saved_model_path",
        type=str,
        default="./pruned_model.pth",
        help="Path to save the pruned model",
    )
    parser.add_argument(
        "--f_epochs",
        type=int,
        default=10,
        help="Number of epochs for fine-tuning",
    )
    parser.add_argument(
        "--f_lr",
        type=float,
        default=0.001,
        help="Learning rate for fine-tuning",
    )
    parser.add_argument(
        "--f_weight_decay",
        type=float,
        default=0.0001,
        help=" weight decay for fine-tuning",
    )
    parser.add_argument(
        "--new_dataset_dir",
        type=str,
        default=None,
        help="Optional new dataset directory for additional testing",
    )

    return parser.parse_args()

def validate_args(args):
    """Check if required files and directories exist"""
    if args.dataset_mode == "hardfake":
        if not os.path.exists(args.hardfake_csv_file):
            raise FileNotFoundError(f"Hardfake CSV file not found: {args.hardfake_csv_file}")
        if not os.path.exists(args.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    elif args.dataset_mode == "rvf10k":
        if not os.path.exists(args.rvf10k_train_csv):
            raise FileNotFoundError(f"RVF10k train CSV file not found: {args.rvf10k_train_csv}")
        if not os.path.exists(args.rvf10k_valid_csv):
            raise FileNotFoundError(f"RVF10k valid CSV file not found: {args.rvf10k_valid_csv}")
        if not os.path.exists(args.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    elif args.dataset_mode == "140k":
        if not os.path.exists(args.realfake140k_train_csv):
            raise FileNotFoundError(f"140k train CSV file not found: {args.realfake140k_train_csv}")
        if not os.path.exists(args.realfake140k_valid_csv):
            raise FileNotFoundError(f"140k valid CSV file not found: {args.realfake140k_valid_csv}")
        if not os.path.exists(args.realfake140k_test_csv):
            raise FileNotFoundError(f"140k test CSV file not found: {args.realfake140k_test_csv}")
        if not os.path.exists(args.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    elif args.dataset_mode == "200k":
        if not os.path.exists(args.realfake200k_train_csv):
            raise FileNotFoundError(f"200k train CSV file not found: {args.realfake200k_train_csv}")
        if not os.path.exists(args.realfake200k_valid_csv):
            raise FileNotFoundError(f"200k valid CSV file not found: {args.realfake200k_valid_csv}")
        if not os.path.exists(args.realfake200k_test_csv):
            raise FileNotFoundError(f"200k test CSV file not found: {args.realfake200k_test_csv}")
        if not os.path.exists(args.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    elif args.dataset_mode == "190k":
        if not os.path.exists(args.realfake190k_root_dir):
            raise FileNotFoundError(f"190k dataset directory not found: {args.realfake190k_root_dir}")
    elif args.dataset_mode == "330k":
        if not os.path.exists(args.realfake330k_root_dir):
            raise FileNotFoundError(f"330k dataset directory not found: {args.realfake330k_root_dir}")

    if args.phase in ["train", "finetune"]:
        if not os.path.exists(args.teacher_ckpt_path):
            raise FileNotFoundError(f"Teacher checkpoint not found: {args.teacher_ckpt_path}")

        if args.arch == "MobileNetV2" and "resnet" in args.teacher_ckpt_path.lower():
            raise ValueError(f"Teacher checkpoint path ({args.teacher_ckpt_path}) is not compatible with MobileNetV2 architecture")
        elif args.arch == "ResNet_50" and "mobilenet" in args.teacher_ckpt_path.lower():
            raise ValueError(f"Teacher checkpoint path ({args.teacher_ckpt_path}) is not compatible with ResNet_50 architecture")

    if args.phase == "finetune" and args.finetune_student_ckpt_path and not os.path.exists(args.finetune_student_ckpt_path):
        raise FileNotFoundError(f"Finetune student checkpoint not found: {args.finetune_student_ckpt_path}")

    if args.phase == "test" and args.sparsed_student_ckpt_path and not os.path.exists(args.sparsed_student_ckpt_path):
        raise FileNotFoundError(f"Sparsed student checkpoint not found: {args.sparsed_student_ckpt_path}")

    if args.new_dataset_dir and not os.path.exists(args.new_dataset_dir):
        raise FileNotFoundError(f"New dataset directory not found: {args.new_dataset_dir}")

def main():
    args = parse_args()
    
    # Validate file and directory existence
    validate_args(args)

    # Set seeds for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Log basic information
    print(f"Running phase: {args.phase}")
    print(f"Dataset mode: {args.dataset_mode}")
    print(f"Device: {args.device}")
    print(f"Architecture: {args.arch}")

    if args.dali:
        print("Using NVIDIA DALI for data loading.")
        from nvidia.dali.pipeline import Pipeline
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator
        # DALI pipeline implementation would go here
    else:
        print("Using standard PyTorch DataLoader.")

    # Execute the corresponding phase
    if args.ddp:
        if args.phase == "train":
            train = TrainDDP(args=args)
            train.main()
        elif args.phase == "finetune":
            finetune = FinetuneDDP(args=args)
            finetune.main()
        elif args.phase == "test":
            test = Test(args=args)
            test.main()
    else:
        if args.phase == "train":
            train = Train(args=args)
            train.main()
        elif args.phase == "finetune":
            finetune = Finetune(args=args)
            finetune.main()
        elif args.phase == "test":
            test = Test(args=args)
            test.main()

if __name__ == "__main__":
    main()

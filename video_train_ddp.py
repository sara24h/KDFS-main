import json
import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
#from data.video_data import Dataset_selector
from model.student.ResNet_sparse_video import (ResNet_50_sparse_uadfv,SoftMaskedConv2d)
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from model.student.GoogleNet_sparse import GoogLeNet_sparse_deepfake
from utils import utils, loss, meter, scheduler
from thop import profile
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.teacher.Mobilenetv2 import MobileNetV2_deepfake
from model.teacher.GoogleNet import GoogLeNet_deepfake
from utils.loss import compute_filter_correlation

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class TrainDDP:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.teacher_ckpt_path = args.teacher_ckpt_path
        self.num_epochs = args.num_epochs
        self.num_frames = getattr(args, 'num_frames', 16)
        self.frame_sampling = getattr(args, 'frame_sampling', 'uniform')
        self.split_ratio = getattr(args, 'split_ratio', (0.7, 0.15, 0.15))
        self.lr = args.lr
        self.warmup_steps = args.warmup_steps
        # کد صحیح:
        self.warmup_start_lr = args.warmup_start_lr
        self.lr_decay_T_max = args.lr_decay_T_max
        self.lr_decay_eta_min = args.lr_decay_eta_min
        self.weight_decay = args.weight_decay
        self.train_batch_size = args.train_batch_size
        self.eval_batch_size = args.eval_batch_size
        self.target_temperature = args.target_temperature
        self.gumbel_start_temperature = args.gumbel_start_temperature
        self.gumbel_end_temperature = args.gumbel_end_temperature
        self.coef_kdloss = args.coef_kdloss
        self.coef_rcloss = args.coef_rcloss
        self.coef_maskloss = args.coef_maskloss
        self.resume = args.resume
        self.start_epoch = 0
        self.best_prec1 = 0
        self.world_size = 0
        self.local_rank = -1
        self.rank = -1

        if self.dataset_mode == "uadfv":
            self.args.dataset_type = "uadfv"
            self.num_classes = 1
            self.image_size = 256
        else:
            raise ValueError("dataset_mode must be 'uadfv' for this script")

        self.arch = args.arch.lower().replace('_', '')
        if self.arch not in ['resnet50', 'mobilenetv2', 'googlenet']:
            raise ValueError(f"Unsupported architecture: '{args.arch}'. "
                             "It must be 'resnet50', 'mobilenetv2', or 'googlenet'.")

    def dist_init(self):
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)

    def result_init(self):
        if self.rank == 0:
            if not os.path.exists(self.result_dir):
                os.makedirs(self.result_dir)
            self.writer = SummaryWriter(self.result_dir)
            self.logger = utils.get_logger(
                os.path.join(self.result_dir, "train_logger.log"), "train_logger"
            )
            self.logger.info("train config:")
            self.logger.info(str(json.dumps(vars(self.args), indent=4)))
            utils.record_config(
                self.args, os.path.join(self.result_dir, "train_config.txt")
            )
            self.logger.info("--------- Train -----------")

    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        self.seed = self.seed + self.rank
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = True

    def dataload(self):
        if self.dataset_mode == "uadfv":
            from data.video_data import create_uadfv_dataloaders

            if self.rank == 0:
                self.logger.info(f"Loading UADFV dataset from: {self.dataset_dir}")
                self.logger.info(f"Number of frames per video: {self.num_frames}")
                self.logger.info(f"Frame sampling strategy: {self.frame_sampling}")
                self.logger.info(f"Split ratio (train/val/test): {self.split_ratio}")

            self.train_loader, self.val_loader, self.test_loader = create_uadfv_dataloaders(
                root_dir=self.dataset_dir,
                num_frames=self.num_frames,
                image_size=self.image_size,
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                ddp=True,
                seed=self.seed,
                sampling_strategy=self.frame_sampling
            )

            # --- شروع بخش جدید: مشخصات میانگین ویدیو ---
            if self.rank == 0:
                self.logger.info("Using pre-calculated average video properties for FLOPs reporting.")
                # مقادیر میانگین برای دیتاست UADFV که از قبل محاسبه کرده‌اید
                self.avg_video_duration = 11.6  # <-- این مقدار را با میانگین واقعی خود جایگزین کنید
                self.avg_video_fps = 30.00      # <-- این مقدار را با میانگین واقعی خود جایگزین کنید
                self.logger.info(f"Average video duration set to: {self.avg_video_duration}s")
                self.logger.info(f"Average video FPS set to: {self.avg_video_fps}")
            # --- پایان بخش جدید ---

            if self.rank == 0:
                self.logger.info("UADFV Dataset has been loaded!")

    def build_model(self):
        if self.rank == 0:
            self.logger.info("==> Building model..")
            self.logger.info("Loading teacher model")

        if self.arch == 'resnet50':
            teacher_model = ResNet_50_hardfakevsreal()
        elif self.arch == 'mobilenetv2':
            teacher_model = MobileNetV2_deepfake()
        elif self.arch == 'googlenet':
            teacher_model = GoogLeNet_deepfake()
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

        ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu")
        state_dict = ckpt_teacher.get('config_state_dict',
                                      ckpt_teacher.get('student', ckpt_teacher))

        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '', 1)
                new_state_dict[name] = v
            state_dict = new_state_dict

        teacher_model.load_state_dict(state_dict, strict=True)
        self.teacher = teacher_model.cuda()

        if self.rank == 0:
            self.logger.info("Building student model")

        if self.arch == 'resnet50':
            StudentModelClass = (ResNet_50_sparse_uadfv
                                 if self.dataset_mode != "hardfake"
                                 else ResNet_50_sparse_hardfakevsreal)
        elif self.arch == 'mobilenetv2':
            StudentModelClass = MobileNetV2_sparse_deepfake
        elif self.arch == 'googlenet':
            StudentModelClass = GoogLeNet_sparse_deepfake
        else:
            raise ValueError(f"Unsupported architecture for student: {self.arch}")

        self.student = StudentModelClass(
            gumbel_start_temperature=self.gumbel_start_temperature,
            gumbel_end_temperature=self.gumbel_end_temperature,
            num_epochs=self.num_epochs,
        )
        self.student.dataset_type = self.args.dataset_type

        if self.arch == 'mobilenetv2':
            num_ftrs = self.student.classifier.in_features
            self.student.classifier = nn.Linear(num_ftrs, 1)
        elif self.arch == 'googlenet':
            num_ftrs = self.student.fc.in_features
            self.student.fc = nn.Linear(num_ftrs, 1)
        else:  # resnet50
            num_ftrs = self.student.fc.in_features
            self.student.fc = nn.Linear(num_ftrs, 1)

        self.student = self.student.cuda()
        self.student = DDP(self.student, device_ids=[self.local_rank])

        # --- DEBUGGING LOG: List all model parameters ---
        if self.rank == 0:
            self.logger.info("--- Student Model Parameters ---")
            for name, param in self.student.module.named_parameters():
                self.logger.info(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
            self.logger.info("--- End of Parameters List ---")

    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss().cuda()
        self.kd_loss = loss.KDLoss().cuda()
        self.rc_loss = loss.RCLoss().cuda()
        self.mask_loss = loss.MaskLoss().cuda()

    def define_optim(self):
        weight_params = []
        mask_params = []

        # --- DEBUGGING LOG: Show which params go to which optimizer ---
        if self.rank == 0:
            self.logger.info("--- Separating Parameters for Optimizers ---")
        
        for name, param in self.student.module.named_parameters():
            if param.requires_grad:
                if "mask" in name.lower():
                    mask_params.append(param)
                    if self.rank == 0:
                        self.logger.info(f"Adding to MASK optimizer: {name}")
                else:
                    weight_params.append(param)
                    if self.rank == 0:
                        self.logger.info(f"Adding to WEIGHT optimizer: {name}")

        if self.rank == 0:
            self.logger.info(f"Found {len(weight_params)} weight parameters and {len(mask_params)} mask parameters.")
            self.logger.info("--- End of Parameter Separation ---")

        self.optim_weight = torch.optim.Adamax(weight_params,
                                              lr=self.lr,
                                              weight_decay=self.weight_decay,
                                              eps=1e-7)

        self.optim_mask = None
        self.scheduler_student_mask = None

        if mask_params:
            self.optim_mask = torch.optim.Adamax(mask_params, lr=self.lr, eps=1e-7)
            self.scheduler_student_mask = scheduler.CosineAnnealingLRWarmup(
                self.optim_mask, T_max=self.lr_decay_T_max,
                eta_min=self.lr_decay_eta_min, last_epoch=-1,
                warmup_steps=self.warmup_steps,
                warmup_start_lr=self.warmup_start_lr)
        elif self.rank == 0:
            self.logger.warning("Warning: No mask parameters found. 'optim_mask' and 'scheduler_student_mask' will be None.")

        self.scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.optim_weight, T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min, last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr)

    def resume_student_ckpt(self):
        if not os.path.exists(self.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume}")

        ckpt_student = torch.load(self.resume, map_location="cpu", weights_only=True)
        self.best_prec1 = ckpt_student["best_prec1"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.module.load_state_dict(ckpt_student["student"])
        self.optim_weight.load_state_dict(ckpt_student["optim_weight"])
        
        if self.optim_mask is not None and "optim_mask" in ckpt_student:
            self.optim_mask.load_state_dict(ckpt_student["optim_mask"])
        elif self.optim_mask is None and "optim_mask" in ckpt_student:
            if self.rank == 0:
                self.logger.warning("Checkpoint contains 'optim_mask' but current model has no mask parameters. Skipping load.")
        
        self.scheduler_student_weight.load_state_dict(ckpt_student["scheduler_student_weight"])

        if self.scheduler_student_mask is not None and "scheduler_student_mask" in ckpt_student:
            self.scheduler_student_mask.load_state_dict(ckpt_student["scheduler_student_mask"])
        elif self.scheduler_student_mask is None and "scheduler_student_mask" in ckpt_student:
            if self.rank == 0:
                self.logger.warning("Checkpoint contains 'scheduler_student_mask' but current model has no mask scheduler. Skipping load.")

        if self.rank == 0:
            self.logger.info(f"=> Continue from epoch {self.start_epoch + 1}...")

    def save_student_ckpt(self, is_best, epoch):
        if self.rank == 0:
            folder = os.path.join(self.result_dir, "student_model")
            if not os.path.exists(folder):
                os.makedirs(folder)

            ckpt_student = {
                "best_prec1": self.best_prec1,
                "start_epoch": epoch,
                "student": self.student.module.state_dict(),
                "optim_weight": self.optim_weight.state_dict(),
                "scheduler_student_weight": self.scheduler_student_weight.state_dict(),
            }

            if self.optim_mask is not None:
                ckpt_student["optim_mask"] = self.optim_mask.state_dict()
            if self.scheduler_student_mask is not None:
                ckpt_student["scheduler_student_mask"] = self.scheduler_student_mask.state_dict()

            if is_best:
                torch.save(ckpt_student,
                           os.path.join(folder, self.arch + "_sparse_best.pt"))
            torch.save(ckpt_student,
                       os.path.join(folder, self.arch + "_sparse_last.pt"))

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def get_mask_averages(self):
        """Get average mask values for each layer"""
        mask_avgs = []
        for m in self.student.module.mask_modules:
            if isinstance(m, SoftMaskedConv2d):
                with torch.no_grad():
                    mask = torch.sigmoid(m.mask_weight)
                    mask_avgs.append(round(mask.mean().item(), 2))
        return mask_avgs

    def train(self):
        if self.rank == 0:
            self.logger.info(f"Starting training from epoch: {self.start_epoch + 1}")

        torch.cuda.empty_cache()
        self.teacher.eval()
        scaler = GradScaler()

        if self.resume:
            self.resume_student_ckpt()

        is_video_dataset = self.dataset_mode in ["uadfv"]

        if self.rank == 0:
            meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
            meter_kdloss = meter.AverageMeter("KDLoss", ":.4e")
            meter_rcloss = meter.AverageMeter("RCLoss", ":.4e")
            meter_maskloss = meter.AverageMeter("MaskLoss", ":.6e")
            meter_loss = meter.AverageMeter("Loss", ":.4e")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
            meter_avg_corr = meter.AverageMeter("L_corr", ":.6f")
            meter_retention = meter.AverageMeter("Retention", ":.4f")

        for epoch in range(self.start_epoch + 1, self.num_epochs + 1):
            self.train_loader.sampler.set_epoch(epoch)
            self.student.train()
            self.student.module.ticket = False

            if self.rank == 0:
                meter_oriloss.reset()
                meter_kdloss.reset()
                meter_rcloss.reset()
                meter_maskloss.reset()
                meter_loss.reset()
                meter_top1.reset()
                meter_avg_corr.reset()
                meter_retention.reset()

                current_lr = self.optim_weight.param_groups[0]['lr']

            self.student.module.update_gumbel_temperature(epoch)
            current_gumbel_temp = self.student.module.gumbel_temperature

            with tqdm(total=len(self.train_loader), ncols=100, disable=self.rank != 0) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description(f"epoch: {epoch}/{self.num_epochs}")

                for batch_data in self.train_loader:
                    self.optim_weight.zero_grad()
                    if self.optim_mask is not None:
                        self.optim_mask.zero_grad()

                    if is_video_dataset:
                        videos, targets = batch_data
                        videos = videos.cuda(non_blocking=True)
                        targets = targets.cuda(non_blocking=True).float()

                        batch_size, num_frames, C, H, W = videos.shape
                        images = videos.view(-1, C, H, W)

                        if torch.isnan(images).any() or torch.isinf(images).any() or \
                           torch.isnan(targets).any() or torch.isinf(targets).any():
                            if self.rank == 0:
                                self.logger.warning("Invalid input detected (NaN or Inf)")
                            continue

                        with autocast():
                            logits_student, feature_list_student = self.student(images)
                            logits_student = logits_student.squeeze(1)
                            logits_student = logits_student.view(batch_size, num_frames).mean(dim=1)

                            with torch.no_grad():
                                logits_teacher, feature_list_teacher = self.teacher(images)
                                logits_teacher = logits_teacher.squeeze(1)
                                logits_teacher = logits_teacher.view(batch_size, num_frames).mean(dim=1)

                            if self.rank == 0 and epoch == 1 and not hasattr(self, '_features_logged'):
                                self._features_logged = True
                                self.logger.info(f"Student features: {len(feature_list_student)} layers")
                                self.logger.info(f"Teacher features: {len(feature_list_teacher)} layers")
                                if len(feature_list_student) > 0:
                                    self.logger.info(f"First student feature shape: {feature_list_student[0].shape}")
                                if len(feature_list_teacher) > 0:
                                    self.logger.info(f"First teacher feature shape: {feature_list_teacher[0].shape}")

                            ori_loss = self.ori_loss(logits_student, targets)
                            kd_loss = (self.target_temperature ** 2) * self.kd_loss(
                                logits_teacher, logits_student, self.target_temperature)

                            rc_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
                            
                            if len(feature_list_student) == 0:
                                if self.rank == 0:
                                    self.logger.warning("Feature list is empty! Model may not be returning features.")
                            else:
                                for i in range(len(feature_list_student)):
                                    layer_rc_loss = self.rc_loss(
                                        feature_list_student[i], 
                                        feature_list_teacher[i]
                                    )
                                    rc_loss = rc_loss + layer_rc_loss
                                
                                rc_loss = rc_loss / len(feature_list_student)
                                
                                if self.rank == 0 and epoch == 1 and not hasattr(self, '_rc_logged'):
                                    self._rc_logged = True
                                    self.logger.info(f"RC Loss per layer: {rc_loss.item():.6f}")

                            mask_loss = self.mask_loss(self.student.module)

                            total_loss = (
                                ori_loss +
                                self.coef_kdloss * kd_loss +
                                self.coef_rcloss * rc_loss +
                                self.coef_maskloss * mask_loss
                            )

                            total_corr, total_ret = 0.0, 0.0
                            n_layers = 0
                            for m in self.student.module.mask_modules:
                                if isinstance(m, SoftMaskedConv2d):
                                    corr, ret = compute_filter_correlation(
                                        m.weight, m.mask_weight,
                                        self.student.module.gumbel_temperature)
                                    total_corr += corr.item()
                                    total_ret += float(ret)
                                    n_layers += 1
                            avg_corr = total_corr / n_layers if n_layers > 0 else 0.0
                            avg_ret = total_ret / n_layers if n_layers > 0 else 0.0

                        scaler.scale(total_loss).backward()
                        scaler.step(self.optim_weight)
                        if self.optim_mask is not None:
                            scaler.step(self.optim_mask)
                        scaler.update()

                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100.0 * correct / batch_size

                    dist.barrier()
                    reduced_ori_loss = self.reduce_tensor(ori_loss.detach())
                    reduced_kd_loss = self.reduce_tensor(kd_loss.detach())
                    reduced_rc_loss = self.reduce_tensor(rc_loss.detach())
                    reduced_mask_loss = self.reduce_tensor(mask_loss.detach())
                    reduced_total_loss = self.reduce_tensor(total_loss.detach())
                    reduced_prec1 = self.reduce_tensor(torch.tensor(prec1, device='cuda'))
                    reduced_avg_corr = self.reduce_tensor(torch.tensor(avg_corr, device='cuda'))
                    reduced_avg_ret = self.reduce_tensor(torch.tensor(avg_ret, device='cuda'))

                    if self.rank == 0:
                        n = batch_size if is_video_dataset else images.size(0)
                        meter_oriloss.update(reduced_ori_loss.item(), n)
                        meter_kdloss.update(self.coef_kdloss * reduced_kd_loss.item(), n)
                        meter_rcloss.update(self.coef_rcloss * reduced_rc_loss.item(), n)
                        meter_maskloss.update(self.coef_maskloss * reduced_mask_loss.item(), n)
                        meter_loss.update(reduced_total_loss.item(), n)
                        meter_top1.update(reduced_prec1.item(), n)
                        meter_avg_corr.update(reduced_avg_corr.item(), n)
                        meter_retention.update(reduced_avg_ret.item(), n)

                        _tqdm.set_postfix(
                            loss=f"{meter_loss.avg:.4f}",
                            acc=f"{meter_top1.avg:.2f}"
                        )
                        _tqdm.update(1)

                if self.rank == 0:
                    self.student.module.ticket = False
                    
                    # --- شروع بخش اصلاح‌شده برای محاسبه FLOPs ویدیو ---
                    avg_video_flops = self.student.module.get_video_flops(
                        video_duration_seconds=self.avg_video_duration, 
                        fps=self.avg_video_fps
                    )
                    # --- پایان بخش اصلاح‌شده ---

                    self.logger.info(f"[Train] Epoch {epoch} : Gumbel_temperature {current_gumbel_temp:.2f} "
                                    f"LR {current_lr:.6f} OriLoss {meter_oriloss.avg:.4f} "
                                    f"KDLoss {meter_kdloss.avg:.4f} RCLoss {meter_rcloss.avg:.6f} "
                                    f"MaskLoss {meter_maskloss.avg:.6f} TotalLoss {meter_loss.avg:.4f} "
                                    f"Train_Acc {meter_top1.avg:.2f}")
                    
                    # لاگ FLOPs بر حسب TFLOPs برای خوانایی بهتر
                    self.logger.info(f"[Train Avg Video Flops] Epoch {epoch} : {avg_video_flops/1e12:.2f} TFLOPs")

            if self.rank == 0:
                self.student.eval()
                self.student.module.ticket = True
                val_meter = meter.AverageMeter("Acc@1", ":6.2f")

                with torch.no_grad():
                    for val_videos, val_targets in self.val_loader:
                        val_videos = val_videos.cuda(non_blocking=True)
                        val_targets = val_targets.cuda(non_blocking=True).float()

                        val_batch_size, val_num_frames, C, H, W = val_videos.shape
                        val_frames = val_videos.view(-1, C, H, W)
                        
                        val_logits, _ = self.student(val_frames)
                        val_logits = val_logits.squeeze(1)
                        val_logits = val_logits.view(val_batch_size, val_num_frames).mean(dim=1)
                        
                        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                        correct = (val_preds == val_targets).sum().item()
                        acc1 = 100.0 * correct / val_batch_size
                        val_meter.update(acc1, val_batch_size)

                mask_avgs = self.get_mask_averages()
                
                # --- شروع بخش اصلاح‌شده برای محاسبه FLOPs ویدیو ---
                val_avg_video_flops = self.student.module.get_video_flops(
                    video_duration_seconds=self.avg_video_duration, 
                    fps=self.avg_video_fps
                )
                # --- پایان بخش اصلاح‌شده ---
                
                self.logger.info(f"[Val] Epoch {epoch} : Val_Acc {val_meter.avg:.2f}")
                self.logger.info(f"[Val mask avg] Epoch {epoch} : {mask_avgs}")
                self.logger.info(f"[Val Avg Video Flops] Epoch {epoch} : {val_avg_video_flops/1e12:.2f} TFLOPs")

                self.scheduler_student_weight.step()
                if self.scheduler_student_mask is not None:
                    self.scheduler_student_mask.step()

                self.writer.add_scalar("train/lr", current_lr, epoch)
                self.writer.add_scalar("train/gumbel_temp", current_gumbel_temp, epoch)
                self.writer.add_scalar("train/acc", meter_top1.avg, epoch)
                self.writer.add_scalar("train/loss", meter_loss.avg, epoch)
                # اضافه کردن FLOPs ویدیو به TensorBoard
                self.writer.add_scalar("train/avg_video_flops", avg_video_flops, epoch)
                self.writer.add_scalar("val/acc", val_meter.avg, epoch)
                self.writer.add_scalar("val/avg_video_flops", val_avg_video_flops, epoch)

                if val_meter.avg > self.best_prec1:
                    self.best_prec1 = val_meter.avg
                    self.logger.info(f" => Best top1 accuracy on validation before finetune : {self.best_prec1}")
                    self.save_student_ckpt(is_best=True, epoch=epoch)
                else:
                    self.save_student_ckpt(is_best=False, epoch=epoch)

        if self.rank == 0:
            self.logger.info("Training finished!")

    def main(self):
        self.dist_init()
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()

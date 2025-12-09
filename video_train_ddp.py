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
from data.video_data import create_uadfv_dataloaders
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv, SoftMaskedConv2d
from utils import utils, loss, meter, scheduler
from thop import profile
from model.teacher.ResNet import ResNet_50_hardfakevsreal

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# دیکشنری FLOPs پایه با مقادیر منطقی برای ویدیو
Flops_baselines = {
    "resnet50": {
        "hardfakevsrealfaces": 7700.0,
        "rvf10k": 5390,
        "140k": 5390.0,
        "200k": 5390.0,
        "330k": 5390.0,
        "190k": 5390.0,
        "uadfv": 172690,  # مقدار منطقی برای 16 فریم
    }
}

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
        self.compress_rate = args.compress_rate
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
        if self.arch not in ['resnet50']:
            raise ValueError(f"Unsupported architecture: '{self.arch}'. It must be 'resnet50'.")

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

    def dataload(self):
        if self.rank == 0:
            self.logger.info(f"Loading UADFV dataset from: {self.dataset_dir}")
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
        if self.rank == 0:
            self.logger.info("UADFV Dataset has been loaded!")

    def build_model(self):
        if self.rank == 0:
            self.logger.info("==> Building model..")
            self.logger.info("Loading teacher model")

        teacher_model = ResNet_50_hardfakevsreal()
        ckpt_teacher = torch.load(self.teacher_ckpt_path, map_location="cpu")
        state_dict = ckpt_teacher.get('config_state_dict', ckpt_teacher.get('student', ckpt_teacher))
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
        self.student = ResNet_50_sparse_uadfv(
            gumbel_start_temperature=self.gumbel_start_temperature,
            gumbel_end_temperature=self.gumbel_end_temperature,
            num_epochs=self.num_epochs,
        )
        self.student.dataset_type = self.args.dataset_type
        num_ftrs = self.student.fc.in_features
        self.student.fc = nn.Linear(num_ftrs, 1)
        self.student = self.student.cuda()
        self.student = DDP(self.student, device_ids=[self.local_rank])

    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss().cuda()
        self.kd_loss = loss.KDLoss().cuda()
        self.rc_loss = loss.RCLoss().cuda()
        self.mask_loss = loss.MaskLoss().cuda()

    def define_optim(self):
        weight_params = []
        mask_params = []
        for name, param in self.student.module.named_parameters():
            if param.requires_grad:
                if "mask" in name.lower():
                    mask_params.append(param)
                else:
                    weight_params.append(param)

        self.optim_weight = torch.optim.Adamax(weight_params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-7)
        self.optim_mask = torch.optim.Adamax(mask_params, lr=self.lr, eps=1e-7)

        self.scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.optim_weight, T_max=self.lr_decay_T_max, eta_min=self.lr_decay_eta_min,
            last_epoch=-1, warmup_steps=self.warmup_steps, warmup_start_lr=self.warmup_start_lr
        )
        self.scheduler_student_mask = scheduler.CosineAnnealingLRWarmup(
            self.optim_mask, T_max=self.lr_decay_T_max, eta_min=self.lr_decay_eta_min,
            last_epoch=-1, warmup_steps=self.warmup_steps, warmup_start_lr=self.warmup_start_lr
        )

    # --- متد جدید برای محاسبه میانگین ماسک‌ها ---
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

        if self.rank == 0:
            meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
            meter_kdloss = meter.AverageMeter("KDLoss", ":.4e")
            meter_rcloss = meter.AverageMeter("RCLoss", ":.4e")
            meter_maskloss = meter.AverageMeter("MaskLoss", ":.6e")
            meter_loss = meter.AverageMeter("Loss", ":.4e")
            meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

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
                lr = self.optim_weight.param_groups[0]["lr"]

            self.student.module.update_gumbel_temperature(epoch)
            with tqdm(total=len(self.train_loader), ncols=100, disable=self.rank != 0) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                for videos, targets in self.train_loader:
                    self.optim_weight.zero_grad()
                    self.optim_mask.zero_grad()
                    videos = videos.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True).float()
                    batch_size, num_frames, C, H, W = videos.shape
                    images = videos.view(-1, C, H, W)

                    with autocast():
                        logits_student, feature_list_student = self.student(images)
                        logits_student = logits_student.squeeze(1)
                        logits_student = logits_student.view(batch_size, num_frames).mean(dim=1)
                        with torch.no_grad():
                            logits_teacher, feature_list_teacher = self.teacher(images)
                            logits_teacher = logits_teacher.squeeze(1)
                            logits_teacher = logits_teacher.view(batch_size, num_frames).mean(dim=1)
                        ori_loss = self.ori_loss(logits_student, targets)
                        kd_loss = (self.target_temperature ** 2) * self.kd_loss(logits_teacher, logits_student, self.target_temperature)
                        rc_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
                        for i in range(len(feature_list_student)):
                            rc_loss = rc_loss + self.rc_loss(feature_list_student[i], feature_list_teacher[i])
                        rc_loss = rc_loss / len(feature_list_student)
                        Flops_baseline = Flops_baselines[self.arch][self.args.dataset_type]
                        current_video_flops = self.student.module.get_video_flops(num_frames=self.num_frames)
                        mask_loss = self.mask_loss(current_video_flops, Flops_baseline * (10**6), self.compress_rate).cuda()
                        total_loss = ori_loss + self.coef_kdloss * kd_loss + self.coef_rcloss * rc_loss + self.coef_maskloss * mask_loss

                    scaler.scale(total_loss).backward()
                    scaler.step(self.optim_weight)
                    scaler.step(self.optim_mask)
                    scaler.update()

                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100. * correct / batch_size

                    dist.barrier()
                    reduced_prec1 = self.reduce_tensor(torch.tensor(prec1).cuda())
                    if self.rank == 0:
                        n = batch_size
                        meter_top1.update(reduced_prec1.item(), n)
                        _tqdm.set_postfix(loss="{:.4f}".format(total_loss.item()), acc="{:.4f}".format(meter_top1.avg))
                        _tqdm.update(1)

            self.scheduler_student_weight.step()
            self.scheduler_student_mask.step()

            # --- شروع بخش لاگ‌گیری آموزش ---
            if self.rank == 0:
                # محاسبه FLOPs و میانگین ماسک‌ها برای آموزش
                train_avg_video_flops = self.student.module.get_video_flops(num_frames=self.num_frames)
                train_mask_avgs = self.get_mask_averages()
                
                self.logger.info(f"[Train Avg Video Flops] Epoch {epoch} : {train_avg_video_flops/1e12:.2f} TFLOPs")
                self.logger.info(f"[Train mask avg] Epoch {epoch} : {train_mask_avgs}")
                self.writer.add_scalar("train/avg_video_flops", train_avg_video_flops, epoch)
            # --- پایان بخش لاگ‌گیری آموزش ---

            # --- شروع بخش اعتبارسنجی ---
            if self.rank == 0:
                self.student.eval()
                self.student.module.ticket = True
                val_meter = meter.AverageMeter("Acc@1", ":6.2f")
                with torch.no_grad():
                    for val_videos, val_targets in self.val_loader:
                        val_videos = val_videos.cuda(non_blocking=True)
                        val_targets = val_targets.cuda(non_blocking=True).float()
                        val_batch_size, val_num_frames, C, H, W = val_videos.shape
                        val_images = val_videos.view(-1, C, H, W)
                        val_logits, _ = self.student(val_images)
                        val_logits = val_logits.squeeze(1)
                        val_logits = val_logits.view(val_batch_size, val_num_frames).mean(dim=1)
                        val_preds = (torch.sigmoid(val_logits) > 0.5).float()
                        correct = (val_preds == val_targets).sum().item()
                        acc1 = 100. * correct / val_batch_size
                        val_meter.update(acc1, val_batch_size)

                # محاسبه FLOPs و میانگین ماسک‌ها برای اعتبارسنجی
                val_avg_video_flops = self.student.module.get_video_flops(num_frames=self.num_frames)
                val_mask_avgs = self.get_mask_averages()
                
                self.logger.info(f"[Val] Epoch {epoch} : Val_Acc {val_meter.avg:.2f}")
                self.logger.info(f"[Val mask avg] Epoch {epoch} : {val_mask_avgs}")
                self.logger.info(f"[Val Avg Video Flops] Epoch {epoch} : {val_avg_video_flops/1e12:.2f} TFLOPs")

                self.writer.add_scalar("val/acc/top1", val_meter.avg, global_step=epoch)
                self.writer.add_scalar("val/avg_video_flops", val_avg_video_flops, epoch)

                if val_meter.avg > self.best_prec1:
                    self.best_prec1 = val_meter.avg
                    self.save_student_ckpt(True, epoch)
                else:
                    self.save_student_ckpt(False, epoch)
            # --- پایان بخش اعتبارسنجی ---

        if self.rank == 0:
            self.logger.info("Training finished!")

    def save_student_ckpt(self, is_best, epoch):
        if self.rank == 0:
            folder = os.path.join(self.result_dir, "student_model")
            if not os.path.exists(folder):
                os.makedirs(folder)
            ckpt_student = {
                "best_prec1": self.best_prec1, "start_epoch": epoch,
                "student": self.student.module.state_dict(),
                "optim_weight": self.optim_weight.state_dict(),
                "optim_mask": self.optim_mask.state_dict(),
                "scheduler_student_weight": self.scheduler_student_weight.state_dict(),
                "scheduler_student_mask": self.scheduler_student_mask.state_dict(),
            }
            if is_best:
                torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_best.pt"))
            torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_last.pt"))

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def main(self):
        self.dist_init()
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()

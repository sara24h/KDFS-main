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
from torch import amp

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

Flops_baselines = {
    "resnet50": {
        "hardfakevsrealfaces": 7700.0,
        "rvf10k": 5390,
        "140k": 5390.0,
        "200k": 5390.0,
        "330k": 5390.0,
        "190k": 5390.0,
        "uadfv": 172690, # مقدار FLOPs برای ویدیو
    }}
}

class TrainDDP:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        # یک بار نام معماری را تمیز و نرمال کنید
        self.arch = args.arch.lower().replace('_', '')
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.teacher_ckpt_path = args.teacher_ckpt_path
        self.num_epochs = args.num_epochs
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

        # اضافه کردن پارامترهای ویدیویی با مقادیر پیش‌فرض
        self.num_frames = getattr(args, 'num_frames', 16)
        self.frame_sampling = getattr(args, 'frame_sampling', 'uniform')
        self.split_ratio = getattr(args, 'split_ratio', (0.7, 0.15, 0.15))

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
            raise ValueError("This script is configured for 'uadfv' dataset only.")

        if self.arch not in ['resnet50']:
            raise ValueError(f"Unsupported architecture: '{self.arch}'. It must be 'resnet50'")
        print("TrainDDP __init__ method executed.") # برای اطمینان از اجرا

    def dist_init(self):
        dist.init_process_group("nccl")
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank()
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        print("dist_init method executed.")

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
        print("result_init method executed.")

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
        print("setup_seed method executed.")

    # تورفتگی این تابع اصلاح شده است
    def dataload(self):
        if self.dataset_mode == "uadfv":
            if self.rank == 0:
                self.logger.info(f"Loading UADFV dataset from: {self.dataset_dir}")
                self.logger.info(f"Number of frames per video: {self.num_frames}")
                self.logger.info(f"Frame sampling strategy: {self.frame_sampling}")

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
        print("dataload method executed.")

    def build_model(self):
        if self.rank == 0:
            self.logger.info("==> Building model..")
            self.logger.info("Loading teacher model")

        if self.arch == 'resnet50':
            teacher_model = ResNet_50_hardfakevsreal()
        
        else:
            raise ValueError(f"Unsupported architecture: {self.arch}")

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

        # منطق ساخت مدل دانش‌آموز ساده‌سازی شده
        if self.arch == 'resnet50':
            StudentModelClass = ResNet_50_sparse_uadfv
        
        else:
            raise ValueError(f"Unsupported architecture for student: {self.arch}")

        self.student = StudentModelClass(
            gumbel_start_temperature=self.gumbel_start_temperature,
            gumbel_end_temperature=self.gumbel_end_temperature,
            num_epochs=self.num_epochs,
        )

        self.student.dataset_type = self.args.dataset_type
        self.student = self.student.cuda()

        num_ftrs = self.student.fc.in_features
        self.student.fc = nn.Linear(num_ftrs, 1).cuda()

        self.student = DDP(self.student, device_ids=[self.local_rank])
        print("build_model method executed.")

    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss().cuda()
        self.kd_loss = loss.KDLoss().cuda()
        self.rc_loss = loss.RCLoss().cuda()
        self.mask_loss = loss.MaskLoss().cuda()
        print("define_loss method executed.")

    def define_optim(self):
        weight_params = []
        mask_params = []
        for name, param in self.student.module.named_parameters():
            if param.requires_grad:
                if "mask" in name.lower():
                    mask_params.append(param)
                else:
                    weight_params.append(param)

        self.optim_weight = torch.optim.Adamax(
            weight_params, lr=self.lr, weight_decay=self.weight_decay, eps=1e-7
        )
        self.optim_mask = torch.optim.Adamax(mask_params, lr=self.lr, eps=1e-7)

        self.scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.optim_weight,
            T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
        )
        self.scheduler_student_mask = scheduler.CosineAnnealingLRWarmup(
            self.optim_mask,
            T_max=self.lr_decay_T_max,
            eta_min=self.lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.warmup_steps,
            warmup_start_lr=self.warmup_start_lr,
        )
        print("define_optim method executed.")

    def resume_student_ckpt(self):
        if not os.path.exists(self.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume}")
        ckpt_student = torch.load(self.resume, map_location="cpu", weights_only=True)
        self.best_prec1 = ckpt_student["best_prec1"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.module.load_state_dict(ckpt_student["student"])
        self.optim_weight.load_state_dict(ckpt_student["optim_weight"])
        self.optim_mask.load_state_dict(ckpt_student["optim_mask"])
        self.scheduler_student_weight.load_state_dict(ckpt_student["scheduler_student_weight"])
        self.scheduler_student_mask.load_state_dict(ckpt_student["scheduler_student_mask"])
        if self.rank == 0:
            self.logger.info("=> Continue from epoch {}...".format(self.start_epoch + 1))
        print("resume_student_ckpt method executed.")

    def save_student_ckpt(self, is_best, epoch):
        if self.rank == 0:
            folder = os.path.join(self.result_dir, "student_model")
            if not os.path.exists(folder):
                os.makedirs(folder)

            ckpt_student = {}
            ckpt_student["best_prec1"] = self.best_prec1
            ckpt_student["start_epoch"] = epoch
            ckpt_student["student"] = self.student.module.state_dict()
            ckpt_student["optim_weight"] = self.optim_weight.state_dict()
            ckpt_student["optim_mask"] = self.optim_mask.state_dict()
            ckpt_student["scheduler_student_weight"] = self.scheduler_student_weight.state_dict()
            ckpt_student["scheduler_student_mask"] = self.scheduler_student_mask.state_dict()

            if is_best:
                torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_best.pt"))
            torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_last.pt"))
        print("save_student_ckpt method executed.")

    def reduce_tensor(self, tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def train(self):
        if self.rank == 0:
            self.logger.info(f"Starting training from epoch: {self.start_epoch + 1}")

        torch.cuda.empty_cache()
        self.teacher.eval()
        scaler = GradScaler()

        if self.resume:
            self.resume_student_ckpt()

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
                lr = (
                    self.optim_weight.state_dict()["param_groups"][0]["lr"]
                    if epoch > 1
                    else self.warmup_start_lr
                )

            self.student.module.update_gumbel_temperature(epoch)
            with tqdm(total=len(self.train_loader), ncols=100, disable=self.rank != 0) as _tqdm:
                if self.rank == 0:
                    _tqdm.set_description("epoch: {}/{}".format(epoch, self.num_epochs))
                
                # --- شروع بخش اصلاح‌شده برای داده‌های ویدیویی ---
                for videos, targets in self.train_loader:
                    self.optim_weight.zero_grad()
                    self.optim_mask.zero_grad()
                    videos = videos.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True).float()
                    
                    batch_size, num_frames, C, H, W = videos.shape
                    # تبدیل ویدیو به توالی فریم
                    images = videos.view(-1, C, H, W)
                    # تکرار برچسب برای هر فریم
                    targets_expanded = targets.unsqueeze(1).repeat(1, num_frames).view(-1)
                    # --- پایان بخش اصلاح‌شده ---

                    if torch.isnan(images).any() or torch.isinf(images).any() or torch.isnan(targets_expanded).any() or torch.isinf(targets_expanded).any():
                        if self.rank == 0:
                            self.logger.warning("Invalid input detected (NaN or Inf)")
                        continue

                    with autocast():
                        logits_student, feature_list_student = self.student(images)
                        logits_student = logits_student.squeeze(1)
                        
                        with torch.no_grad():
                            logits_teacher, feature_list_teacher = self.teacher(images)
                            logits_teacher = logits_teacher.squeeze(1)

                        # --- شروع بخش اصلاح‌شده برای محاسبه زیان در سطح ویدیو ---
                        logits_student = logits_student.view(batch_size, num_frames).mean(dim=1)
                        logits_teacher = logits_teacher.view(batch_size, num_frames).mean(dim=1)
                        
                        ori_loss = self.ori_loss(logits_student, targets)
                        kd_loss = (self.target_temperature**2) * self.kd_loss(
                            logits_teacher, logits_student, self.target_temperature
                        )
                        # --- پایان بخش اصلاح‌شده ---

                        rc_loss = torch.tensor(0.0, device=images.device, dtype=torch.float32)
                        for i in range(len(feature_list_student)):
                            rc_loss = rc_loss + self.rc_loss(
                                feature_list_student[i], feature_list_teacher[i]
                            )

                        Flops_baseline = Flops_baselines[self.arch][self.args.dataset_type]
                        Flops = self.student.module.get_flops()
                        mask_loss = self.mask_loss(
                            Flops, Flops_baseline * (10**6), self.compress_rate
                        ).cuda()

                        total_loss = (
                            ori_loss
                            + self.coef_kdloss * kd_loss
                            + self.coef_rcloss * rc_loss / len(feature_list_student)
                            + self.coef_maskloss * mask_loss
                        )

                    scaler.scale(total_loss).backward()
                    scaler.step(self.optim_weight)
                    scaler.step(self.optim_mask)
                    scaler.update()

                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100. * correct / batch_size # محاسبه دقت بر اساس اندازه دسته ویدیو

                    dist.barrier()
                    reduced_ori_loss = self.reduce_tensor(ori_loss)
                    reduced_kd_loss = self.reduce_tensor(kd_loss)
                    reduced_rc_loss = self.reduce_tensor(rc_loss)
                    reduced_mask_loss = self.reduce_tensor(mask_loss)
                    reduced_total_loss = self.reduce_tensor(total_loss)
                    reduced_prec1 = self.reduce_tensor(torch.tensor(prec1).cuda())

                    if self.rank == 0:
                        n = batch_size
                        meter_oriloss.update(reduced_ori_loss.item(), n)
                        meter_kdloss.update(self.coef_kdloss * reduced_kd_loss.item(), n)
                        meter_rcloss.update(
                            self.coef_rcloss * reduced_rc_loss.item() / len(feature_list_student), n
                        )
                        meter_maskloss.update(self.coef_maskloss * reduced_mask_loss.item(), n)
                        meter_loss.update(reduced_total_loss.item(), n)
                        meter_top1.update(reduced_prec1.item(), n)

                        _tqdm.set_postfix(
                            loss="{:.4f}".format(meter_loss.avg),
                            train_acc="{:.4f}".format(meter_top1.avg),
                        )
                        _tqdm.update(1)

                    time.sleep(0.01)

            self.scheduler_student_weight.step()
            self.scheduler_student_mask.step()

            if self.rank == 0:
                Flops = self.student.module.get_flops()
                self.writer.add_scalar("train/loss/ori_loss", meter_oriloss.avg, global_step=epoch)
                self.writer.add_scalar("train/loss/kd_loss", meter_kdloss.avg, global_step=epoch)
                self.writer.add_scalar("train/loss/rc_loss", meter_rcloss.avg, global_step=epoch)
                self.writer.add_scalar("train/loss/mask_loss", meter_maskloss.avg, global_step=epoch)
                self.writer.add_scalar("train/loss/total_loss", meter_loss.avg, global_step=epoch)
                self.writer.add_scalar("train/acc/top1", meter_top1.avg, global_step=epoch)
                self.writer.add_scalar("train/lr/lr", lr, global_step=epoch)
                self.writer.add_scalar("train/temperature/gumbel_temperature", self.student.module.gumbel_temperature, global_step=epoch)
                self.writer.add_scalar("train/Flops", Flops, global_step=epoch)

                self.logger.info(
                    "[Train] "
                    "Epoch {0} : "
                    "Gumbel_temperature {gumbel_temperature:.2f} "
                    "LR {lr:.6f} "
                    "OriLoss {ori_loss:.4f} "
                    "KDLoss {kd_loss:.4f} "
                    "RCLoss {rc_loss:.4f} "
                    "MaskLoss {mask_loss:.6f} "
                    "TotalLoss {total_loss:.4f} "
                    "Train_Acc {train_acc:.2f}".format(
                        epoch,
                        gumbel_temperature=self.student.module.gumbel_temperature,
                        lr=lr,
                        ori_loss=meter_oriloss.avg,
                        kd_loss=meter_kdloss.avg,
                        rc_loss=meter_rcloss.avg,
                        mask_loss=meter_maskloss.avg,
                        total_loss=meter_loss.avg,
                        train_acc=meter_top1.avg,
                    )
                )

                masks = []
                for _, m in enumerate(self.student.module.mask_modules):
                    masks.append(round(m.mask.mean().item(), 2))
                self.logger.info("[Train mask avg] Epoch {0} : ".format(epoch) + str(masks))

                self.logger.info(
                    "[Train model Flops] Epoch {0} : ".format(epoch)
                    + str(Flops.item() / (10**6))
                    + "M"
                )

            # Validation
            if self.rank == 0:
                self.student.eval()
                self.student.module.ticket = True
                meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")
                with torch.no_grad():
                    with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                        _tqdm.set_description("Validation epoch: {}/{}".format(epoch, self.num_epochs))
                        # --- شروع بخش اصلاح‌شده برای اعتبارسنجی ویدیویی ---
                        for val_videos, val_targets in self.val_loader:
                            val_videos = val_videos.cuda(non_blocking=True)
                            val_targets = val_targets.cuda(non_blocking=True).float()
                            
                            val_batch_size, val_num_frames, C, H, W = val_videos.shape
                            val_images = val_videos.view(-1, C, H, W)
                            
                            logits_student, _ = self.student(val_images)
                            logits_student = logits_student.squeeze(1)
                            logits_student = logits_student.view(val_batch_size, val_num_frames).mean(dim=1)
                            # --- پایان بخش اصلاح‌شده ---
                            
                            preds = (torch.sigmoid(logits_student) > 0.5).float()
                            correct = (preds == val_targets).sum().item()
                            prec1 = 100. * correct / val_batch_size
                            n = val_batch_size
                            meter_top1.update(prec1, n)

                            _tqdm.set_postfix(
                                val_acc="{:.4f}".format(meter_top1.avg),
                            )
                            _tqdm.update(1)
                            time.sleep(0.01)

                Flops = self.student.module.get_flops()
                self.writer.add_scalar("val/acc/top1", meter_top1.avg, global_step=epoch)
                self.writer.add_scalar("val/Flops", Flops, global_step=epoch)

                self.logger.info(
                    "[Val] "
                    "Epoch {0} : "
                    "Val_Acc {val_acc:.2f}".format(
                        epoch,
                        val_acc=meter_top1.avg,
                    )
                )

                masks = []
                for _, m in enumerate(self.student.module.mask_modules):
                    masks.append(round(m.mask.mean().item(), 2))
                self.logger.info("[Val mask avg] Epoch {0} : ".format(epoch) + str(masks))

                self.logger.info(
                    "[Val model Flops] Epoch {0} : ".format(epoch)
                    + str(Flops.item() / (10**6))
                    + "M"
                )

                if self.best_prec1 < meter_top1.avg:
                    self.best_prec1 = meter_top1.avg
                    self.save_student_ckpt(True, epoch)
                else:
                    self.save_student_ckpt(False, epoch)

                self.logger.info(
                    " => Best top1 accuracy on validation before finetune : " + str(self.best_prec1)
                )

        if self.rank == 0:
            self.logger.info("Train finished!")

    def main(self):
        self.dist_init()
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.train()
        print("main method finished.")

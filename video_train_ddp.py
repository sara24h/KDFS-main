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
from data.dataset import Dataset_selector
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv, SoftMaskedConv2d
from model.student.MobileNetV2_sparse import MobileNetV2_sparse_deepfake
from utils import utils, loss, meter, scheduler
from thop import profile
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.teacher.MobilenetV2 import MobileNetV2_deepfake
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
        "uadfv": 172690,
        
    },
    "mobilenetv2": {
        "hardfakevsrealfaces": 7700.0,
        "rvf10k": 416.68,
        "140k": 416.68,
        "200k": 416.68,
        "330k": 416.68,
        "190k": 416.68,
        
    }
}

class TrainDDP:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
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

        self.dali = args.dali

        self.start_epoch = 0
        self.best_prec1 = 0

        self.world_size = 0
        self.local_rank = -1
        self.rank = -1

        if self.dataset_mode == "hardfake":
            self.args.dataset_type = "hardfakevsrealfaces"
            self.num_classes = 1
            self.image_size = 300
        elif self.dataset_mode == "rvf10k":
            self.args.dataset_type = "rvf10k"
            self.num_classes = 1
            self.image_size = 256
        elif self.dataset_mode == "140k":
            self.args.dataset_type = "140k"
            self.num_classes = 1
            self.image_size = 256
        elif self.dataset_mode == "200k":
            self.args.dataset_type = "200k"
            self.num_classes = 1
            self.image_size = 256
        elif self.dataset_mode == "190k":
            self.args.dataset_type = "190k"
            self.num_classes = 1
            self.image_size = 256
        elif self.dataset_mode == "330k":
            self.args.dataset_type = "330k"
            self.num_classes = 1
            self.image_size = 256
        else:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', '140k', '200k', '190k', or '330k'")

        self.arch = args.arch.lower().replace('_', '')

        if self.arch not in ['resnet50', 'mobilenetv2']:
            raise ValueError(f"Unsupported architecture: '{args.arch}'. It must be 'resnet50' or 'MobileNetV2'.")

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
        if self.dataset_mode not in ['hardfake', 'rvf10k', '140k', '200k', '190k', '330k']:
            raise ValueError("dataset_mode must be 'hardfake', 'rvf10k', '140k', '200k', '190k', or '330k'")

        hardfake_csv_file = None
        hardfake_root_dir = None
        rvf10k_train_csv = None
        rvf10k_valid_csv = None
        rvf10k_root_dir = None
        realfake140k_train_csv = None
        realfake140k_valid_csv = None
        realfake140k_test_csv = None
        realfake140k_root_dir = None
        realfake200k_train_csv = None
        realfake200k_val_csv = None
        realfake200k_test_csv = None
        realfake200k_root_dir = None
        realfake190k_root_dir = None
        realfake330k_root_dir = None

        if self.dataset_mode == 'hardfake':
            hardfake_csv_file = os.path.join(self.dataset_dir, 'data.csv')
            hardfake_root_dir = self.dataset_dir
            if self.rank == 0 and not os.path.exists(hardfake_csv_file):
                raise FileNotFoundError(f"CSV file not found: {hardfake_csv_file}")
        elif self.dataset_mode == 'rvf10k':
            rvf10k_train_csv = os.path.join(self.dataset_dir, 'train.csv')
            rvf10k_valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
            rvf10k_root_dir = self.dataset_dir
            if self.rank == 0:
                if not os.path.exists(rvf10k_train_csv):
                    raise FileNotFoundError(f"Train CSV file not found: {rvf10k_train_csv}")
                if not os.path.exists(rvf10k_valid_csv):
                    raise FileNotFoundError(f"Valid CSV file not found: {rvf10k_valid_csv}")
        elif self.dataset_mode == '140k':
            realfake140k_train_csv = os.path.join(self.dataset_dir, 'train.csv')
            realfake140k_valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
            realfake140k_test_csv = os.path.join(self.dataset_dir, 'test.csv')
            realfake140k_root_dir = self.dataset_dir
            if self.rank == 0:
                if not os.path.exists(realfake140k_train_csv):
                    raise FileNotFoundError(f"Train CSV file not found: {realfake140k_train_csv}")
                if not os.path.exists(realfake140k_valid_csv):
                    raise FileNotFoundError(f"Valid CSV file not found: {realfake140k_valid_csv}")
                if not os.path.exists(realfake140k_test_csv):
                    raise FileNotFoundError(f"Test CSV file not found: {realfake140k_test_csv}")
        elif self.dataset_mode == '200k':
            realfake200k_train_csv = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/train_labels.csv"
            realfake200k_val_csv = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/val_labels.csv"
            realfake200k_test_csv = "/kaggle/input/200k-real-vs-ai-visuals-by-mbilal/test_labels.csv"
            realfake200k_root_dir = self.dataset_dir
            if self.rank == 0:
                if not os.path.exists(realfake200k_train_csv):
                    raise FileNotFoundError(f"Train CSV file not found: {realfake200k_train_csv}")
                if not os.path.exists(realfake200k_val_csv):
                    raise FileNotFoundError(f"Valid CSV file not found: {realfake200k_val_csv}")
                if not os.path.exists(realfake200k_test_csv):
                    raise FileNotFoundError(f"Test CSV file not found: {realfake200k_test_csv}")
        elif self.dataset_mode == '190k':
            realfake190k_root_dir = self.dataset_dir
            if self.rank == 0 and not os.path.exists(realfake190k_root_dir):
                raise FileNotFoundError(f"190k dataset directory not found: {realfake190k_root_dir}")
        elif self.dataset_mode == '330k':
            realfake330k_root_dir = self.dataset_dir
            if self.rank == 0 and not os.path.exists(realfake330k_root_dir):
                raise FileNotFoundError(f"330k dataset directory not found: {realfake330k_root_dir}")

        if self.rank == 0:
            self.logger.info(f"Loading dataset: {self.dataset_mode}")

        dataset_instance = Dataset_selector(
            dataset_mode=self.dataset_mode,
            hardfake_csv_file=hardfake_csv_file,
            hardfake_root_dir=hardfake_root_dir,
            rvf10k_train_csv=rvf10k_train_csv,
            rvf10k_valid_csv=rvf10k_valid_csv,
            rvf10k_root_dir=rvf10k_root_dir,
            realfake140k_train_csv=realfake140k_train_csv,
            realfake140k_valid_csv=realfake140k_valid_csv,
            realfake140k_test_csv=realfake140k_test_csv,
            realfake140k_root_dir=realfake140k_root_dir,
            realfake200k_train_csv=realfake200k_train_csv,
            realfake200k_val_csv=realfake200k_val_csv,
            realfake200k_test_csv=realfake200k_test_csv,
            realfake200k_root_dir=realfake200k_root_dir,
            realfake190k_root_dir=realfake190k_root_dir,
            realfake330k_root_dir=realfake330k_root_dir,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            ddp=True
        )

        self.train_loader = dataset_instance.loader_train
        self.val_loader = dataset_instance.loader_val
        self.test_loader = dataset_instance.loader_test
        if self.rank == 0:
            self.logger.info("Dataset has been loaded!")

    def build_model(self):
        if self.rank == 0:
            self.logger.info("==> Building model..")
            self.logger.info("Loading teacher model")

        if self.arch == 'resnet50':
            teacher_model = ResNet_50_hardfakevsreal()
        elif self.arch == 'mobilenetv2':
            teacher_model = MobileNetV2_deepfake()
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
            self.logger.info("Testing teacher model on validation batch...")
            with torch.no_grad():
                correct, total = 0, 0
                for images, targets in self.val_loader:
                    images, targets = images.cuda(), targets.cuda().float()
                    logits, _ = self.teacher(images)
                    logits = logits.squeeze(1)
                    preds = (torch.sigmoid(logits) > 0.5).float()
                    correct += (preds == targets).sum().item()
                    total += images.size(0)
                    break 
                accuracy = 100. * correct / total
                self.logger.info(f"Teacher accuracy on validation batch: {accuracy:.2f}%")

        if self.rank == 0:
            self.logger.info("Building student model")

        if self.arch.lower() == 'resnet50':
            StudentModelClass = ResNet_50_sparse_rvf10k if self.dataset_mode != "hardfake" else ResNet_50_sparse_hardfakevsreal
        elif self.arch.lower() == 'mobilenetv2':
            StudentModelClass = MobileNetV2_sparse_deepfake
        else:
            raise ValueError(f"Unsupported architecture for student: {self.arch}")

        self.student = StudentModelClass(
            gumbel_start_temperature=self.gumbel_start_temperature,
            gumbel_end_temperature=self.gumbel_end_temperature,
            num_epochs=self.num_epochs,
        )

        self.student.dataset_type = self.args.dataset_type
        
        self.student = self.student.cuda()

        if self.arch.lower() == 'mobilenetv2':
            num_ftrs = self.student.classifier.in_features
            self.student.classifier = nn.Linear(num_ftrs, 1).cuda()
        else:  
            num_ftrs = self.student.fc.in_features
            self.student.fc = nn.Linear(num_ftrs, 1).cuda()

        self.student = DDP(self.student, device_ids=[self.local_rank])

    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss().cuda()
        self.kd_loss = loss.KDLoss().cuda()
        self.rc_loss = loss.RCLoss().cuda()
        self.mask_loss = loss.MaskLoss().cuda()

    def define_optim(self):
        weight_params = map(
            lambda a: a[1],
            filter(
                lambda p: p[1].requires_grad and "mask" not in p[0],
                self.student.module.named_parameters(),
            ),
        )
        mask_params = map(
            lambda a: a[1],
            filter(
                lambda p: p[1].requires_grad and "mask" in p[0],
                self.student.module.named_parameters(),
            ),
        )

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

    def resume_student_ckpt(self):
        if not os.path.exists(self.resume):
            raise FileNotFoundError(f"Checkpoint file not found: {self.resume}")
        ckpt_student = torch.load(self.resume, map_location="cpu", weights_only=True)
        self.best_prec1 = ckpt_student["best_prec1"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.module.load_state_dict(ckpt_student["student"])
        self.optim_weight.load_state_dict(ckpt_student["optim_weight"])
        self.optim_mask.load_state_dict(ckpt_student["optim_mask"])
        self.scheduler_student_weight.load_state_dict(
            ckpt_student["scheduler_student_weight"]
        )
        self.scheduler_student_mask.load_state_dict(
            ckpt_student["scheduler_student_mask"]
        )
        if self.rank == 0:
            self.logger.info("=> Continue from epoch {}...".format(self.start_epoch + 1))

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
                torch.save(
                    ckpt_student,
                    os.path.join(folder, self.arch + "_sparse_best.pt"),
                )
            torch.save(ckpt_student, os.path.join(folder, self.arch + "_sparse_last.pt"))

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
                for images, targets in self.train_loader:
                    self.optim_weight.zero_grad()
                    self.optim_mask.zero_grad()
                    images = images.cuda()
                    targets = targets.cuda().float()

                    if torch.isnan(images).any() or torch.isinf(images).any() or torch.isnan(targets).any() or torch.isinf(targets).any():
                        if self.rank == 0:
                            self.logger.warning("Invalid input detected (NaN or Inf)")
                        continue

                    with autocast():
                        logits_student, feature_list_student = self.student(images)
                        logits_student = logits_student.squeeze(1)
                        with torch.no_grad():
                            logits_teacher, feature_list_teacher = self.teacher(images)
                            logits_teacher = logits_teacher.squeeze(1)

                        ori_loss = self.ori_loss(logits_student, targets)

                        kd_loss = (self.target_temperature**2) * self.kd_loss(
                            logits_teacher,
                            logits_student,
                            self.target_temperature
                        )

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
                    prec1 = 100. * correct / images.size(0)

                    dist.barrier()
                    reduced_ori_loss = self.reduce_tensor(ori_loss)
                    reduced_kd_loss = self.reduce_tensor(kd_loss)
                    reduced_rc_loss = self.reduce_tensor(rc_loss)
                    reduced_mask_loss = self.reduce_tensor(mask_loss)
                    reduced_total_loss = self.reduce_tensor(total_loss)
                    reduced_prec1 = self.reduce_tensor(torch.tensor(prec1).cuda())

                    if self.rank == 0:
                        n = images.size(0)
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
                        for images, targets in self.val_loader:
                            images = images.cuda()
                            targets = targets.cuda().float()

                            if torch.isnan(images).any() or torch.isinf(images).any() or torch.isnan(targets).any() or torch.isinf(targets).any():
                                self.logger.warning("Invalid input detected in validation (NaN or Inf)")
                                continue

                            logits_student, _ = self.student(images)
                            logits_student = logits_student.squeeze(1)
                            preds = (torch.sigmoid(logits_student) > 0.5).float()
                            correct = (preds == targets).sum().item()
                            prec1 = 100. * correct / images.size(0)
                            n = images.size(0)
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

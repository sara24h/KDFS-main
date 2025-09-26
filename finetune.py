import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
import numpy as np
from tqdm import tqdm
import time
from utils import utils, loss, meter, scheduler
from data.dataset import Dataset_selector
from model.student.ResNet_sparse import ResNet_50_sparse_hardfakevsreal, ResNet_50_sparse_rvf10k

class Finetune:
    def __init__(self, args):
        self.args = args
        self.dataset_dir = args.dataset_dir
        self.dataset_mode = args.dataset_mode
        if self.dataset_mode == "hardfake":
            self.dataset_type = "hardfakevsrealfaces"
        elif self.dataset_mode == "rvf10k":
            self.dataset_type = "rvf10k"
        elif self.dataset_mode == "140k":
            self.dataset_type = "140k"
        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.arch = args.arch
        self.device = args.device
        self.seed = args.seed
        self.result_dir = args.result_dir
        self.finetune_train_batch_size = args.finetune_train_batch_size
        self.finetune_eval_batch_size = args.finetune_eval_batch_size
        self.finetune_student_ckpt_path = args.finetune_student_ckpt_path
        self.finetune_num_epochs = args.finetune_num_epochs
        self.finetune_lr = args.finetune_lr
        self.finetune_warmup_steps = args.finetune_warmup_steps
        self.finetune_warmup_start_lr = args.finetune_warmup_start_lr
        self.finetune_lr_decay_T_max = args.finetune_lr_decay_T_max
        self.finetune_lr_decay_eta_min = args.finetune_lr_decay_eta_min
        self.finetune_weight_decay = args.finetune_weight_decay
        self.finetune_resume = args.finetune_resume
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path

        self.start_epoch = 0
        self.best_prec1_after_finetune = 0

    def result_init(self):
        self.writer = SummaryWriter(self.result_dir)
        self.logger = utils.get_logger(
            os.path.join(self.result_dir, "finetune_logger.log"), "finetune_logger"
        )
        self.logger.info("finetune config:")
        self.logger.info(str(json.dumps(vars(self.args), indent=4)))
        utils.record_config(
            self.args, os.path.join(self.result_dir, "finetune_config.txt")
        )
        self.logger.info("--------- Finetune -----------")

    def setup_seed(self):
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # برای کاهش خطاهای CUDA
        torch.use_deterministic_algorithms(True)
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
        if self.dataset_mode == 'hardfake':
            hardfake_csv_file = os.path.join(self.dataset_dir, 'data.csv')
            hardfake_root_dir = self.dataset_dir
            dataset = Dataset_selector(
                dataset_mode='hardfake',
                hardfake_csv_file=hardfake_csv_file,
                hardfake_root_dir=hardfake_root_dir,
                train_batch_size=self.finetune_train_batch_size,
                eval_batch_size=self.finetune_eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        elif self.dataset_mode == 'rvf10k':
            rvf10k_train_csv = os.path.join(self.dataset_dir, 'train.csv')
            rvf10k_valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
            rvf10k_root_dir = self.dataset_dir
            dataset = Dataset_selector(
                dataset_mode='rvf10k',
                rvf10k_train_csv=rvf10k_train_csv,
                rvf10k_valid_csv=rvf10k_valid_csv,
                rvf10k_root_dir=rvf10k_root_dir,
                train_batch_size=self.finetune_train_batch_size,
                eval_batch_size=self.finetune_eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        elif self.dataset_mode == '140k':
            realfake140k_train_csv = os.path.join(self.dataset_dir, 'train.csv')
            realfake140k_valid_csv = os.path.join(self.dataset_dir, 'valid.csv')
            realfake140k_test_csv = os.path.join(self.dataset_dir, 'test.csv')
            realfake140k_root_dir = self.dataset_dir
            dataset = Dataset_selector(
                dataset_mode='140k',
                realfake140k_train_csv=realfake140k_train_csv,
                realfake140k_valid_csv=realfake140k_valid_csv,
                realfake140k_test_csv=realfake140k_test_csv,
                realfake140k_root_dir=realfake140k_root_dir,
                train_batch_size=self.finetune_train_batch_size,
                eval_batch_size=self.finetune_eval_batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        else:
            raise ValueError(f"Unknown dataset_mode: {self.dataset_mode}")

        self.train_loader = dataset.loader_train
        self.val_loader = dataset.loader_val
        self.test_loader = dataset.loader_test
        self.logger.info("Dataset has been loaded!")

    def build_model(self):
        self.logger.info("==> Building model..")
        self.logger.info("Loading student model")
        if not os.path.exists(self.finetune_student_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {self.finetune_student_ckpt_path}")
        if self.dataset_mode == "hardfake":
            self.student = ResNet_50_sparse_hardfakevsreal()
        else:  # rvf10k or 140k
            self.student = ResNet_50_sparse_rvf10k()
        ckpt_student = torch.load(self.finetune_student_ckpt_path, map_location="cpu", weights_only=True)
        self.student.load_state_dict(ckpt_student["student"])
        self.best_prec1_before_finetune = ckpt_student["best_prec1"]
        self.student = self.student.to(self.device)

    def define_loss(self):
        self.ori_loss = nn.BCEWithLogitsLoss()  # برای مسئله باینری

    def define_optim(self):
        weight_params = map(
            lambda a: a[1],
            filter(
                lambda p: p[1].requires_grad and "mask" not in p[0],
                self.student.named_parameters(),
            ),
        )
        self.finetune_optim_weight = torch.optim.Adamax(
            weight_params,
            lr=self.finetune_lr,
            weight_decay=self.finetune_weight_decay,
            eps=1e-7,
        )
        self.finetune_scheduler_student_weight = scheduler.CosineAnnealingLRWarmup(
            self.finetune_optim_weight,
            T_max=self.finetune_lr_decay_T_max,
            eta_min=self.finetune_lr_decay_eta_min,
            last_epoch=-1,
            warmup_steps=self.finetune_warmup_steps,
            warmup_start_lr=self.finetune_warmup_start_lr,
        )

    def resume_student_ckpt(self):
        ckpt_student = torch.load(self.finetune_resume, map_location="cpu", weights_only=True)
        self.best_prec1_after_finetune = ckpt_student["best_prec1_after_finetune"]
        self.start_epoch = ckpt_student["start_epoch"]
        self.student.load_state_dict(ckpt_student["student"])
        self.finetune_optim_weight.load_state_dict(
            ckpt_student["finetune_optim_weight"]
        )
        self.finetune_scheduler_student_weight.load_state_dict(
            ckpt_student["finetune_scheduler_student_weight"]
        )
        self.logger.info(f"=> Continue from epoch {self.start_epoch}...")

    def save_student_ckpt(self, is_best):
        folder = os.path.join(self.result_dir, "student_model")
        if not os.path.exists(folder):
            os.makedirs(folder)

        ckpt_student = {
            "best_prec1_after_finetune": self.best_prec1_after_finetune,
            "start_epoch": self.start_epoch,
            "student": self.student.state_dict(),
            "finetune_optim_weight": self.finetune_optim_weight.state_dict(),
            "finetune_scheduler_student_weight": self.finetune_scheduler_student_weight.state_dict(),
        }

        if is_best:
            torch.save(
                ckpt_student,
                os.path.join(folder, f"finetune_{self.arch}_sparse_best.pt"),
            )
        torch.save(
            ckpt_student,
            os.path.join(folder, f"finetune_{self.arch}_sparse_last.pt"),
        )

    def finetune(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            self.student = self.student.cuda()
            self.ori_loss = self.ori_loss.cuda()
        if self.finetune_resume:
            self.resume_student_ckpt()

        meter_oriloss = meter.AverageMeter("OriLoss", ":.4e")
        meter_loss = meter.AverageMeter("Loss", ":.4e")
        meter_top1 = meter.AverageMeter("Acc@1", ":6.2f")

        for epoch in range(self.start_epoch + 1, self.finetune_num_epochs + 1):
            # آموزش
            self.student.train()
            self.student.ticket = True
            meter_oriloss.reset()
            meter_loss.reset()
            meter_top1.reset()
            finetune_lr = (
                self.finetune_optim_weight.state_dict()["param_groups"][0]["lr"]
                if epoch > 1
                else self.finetune_warmup_start_lr
            )

            with tqdm(total=len(self.train_loader), ncols=100) as _tqdm:
                _tqdm.set_description(f"epoch: {epoch}/{self.finetune_num_epochs}")
                for images, targets in self.train_loader:
                    self.finetune_optim_weight.zero_grad()
                    if self.device == "cuda":
                        images = images.cuda()
                        targets = targets.cuda().float()
                    logits_student, _ = self.student(images)
                    logits_student = logits_student.squeeze(1)  # برای مسئله باینری
                    ori_loss = self.ori_loss(logits_student, targets)
                    total_loss = ori_loss
                    total_loss.backward()
                    self.finetune_optim_weight.step()

                    preds = (torch.sigmoid(logits_student) > 0.5).float()
                    correct = (preds == targets).sum().item()
                    prec1 = 100. * correct / images.size(0)
                    n = images.size(0)
                    meter_oriloss.update(ori_loss.item(), n)
                    meter_loss.update(total_loss.item(), n)
                    meter_top1.update(prec1, n)

                    _tqdm.set_postfix(
                        loss=f"{meter_loss.avg:.4f}",
                        top1=f"{meter_top1.avg:.4f}",
                    )
                    _tqdm.update(1)
                    time.sleep(0.01)

            self.finetune_scheduler_student_weight.step()

            self.writer.add_scalar("finetune_train/loss/ori_loss", meter_oriloss.avg, epoch)
            self.writer.add_scalar("finetune_train/loss/total_loss", meter_loss.avg, epoch)
            self.writer.add_scalar("finetune_train/acc/top1", meter_top1.avg, epoch)
            self.writer.add_scalar("finetune_train/lr/lr", finetune_lr, epoch)

            self.logger.info(
                f"[Finetune_train] Epoch {epoch} : "
                f"LR {finetune_lr:.6f} "
                f"OriLoss {meter_oriloss.avg:.4f} "
                f"TotalLoss {meter_loss.avg:.4f} "
                f"Prec@1 {meter_top1.avg:.2f}"
            )

            # اعتبارسنجی
            self.student.eval()
            self.student.ticket = True
            meter_top1.reset()
            with torch.no_grad():
                with tqdm(total=len(self.val_loader), ncols=100) as _tqdm:
                    _tqdm.set_description(f"epoch: {epoch}/{self.finetune_num_epochs}")
                    for images, targets in self.val_loader:
                        if self.device == "cuda":
                            images = images.cuda()
                            targets = targets.cuda().float()
                        logits_student, _ = self.student(images)
                        logits_student = logits_student.squeeze(1)
                        preds = (torch.sigmoid(logits_student) > 0.5).float()
                        correct = (preds == targets).sum().item()
                        prec1 = 100. * correct / images.size(0)
                        n = images.size(0)
                        meter_top1.update(prec1, n)
                        _tqdm.set_postfix(top1=f"{meter_top1.avg:.4f}")
                        _tqdm.update(1)
                        time.sleep(0.01)

            self.writer.add_scalar("finetune_val/acc/top1", meter_top1.avg, epoch)
            self.logger.info(
                f"[Finetune_val] Epoch {epoch} : Prec@1 {meter_top1.avg:.2f}"
            )

            masks = [round(m.mask.mean().item(), 2) for m in self.student.mask_modules]
            self.logger.info(f"[Mask avg] Epoch {epoch} : {masks}")

            self.start_epoch += 1
            if self.best_prec1_after_finetune < meter_top1.avg:
                self.best_prec1_after_finetune = meter_top1.avg
                self.save_student_ckpt(True)
            else:
                self.save_student_ckpt(False)

            self.logger.info(f" => Best top1 accuracy before finetune : {self.best_prec1_before_finetune}")
            self.logger.info(f" => Best top1 accuracy after finetune : {self.best_prec1_after_finetune}")

        self.logger.info("Finetune finished!")
        self.logger.info(f"Best top1 accuracy : {self.best_prec1_after_finetune}")

        # محاسبه و لاگ کردن FLOPs و Params
        try:
            (
                Flops_baseline,
                Flops,
                Flops_reduction,
                Params_baseline,
                Params,
                Params_reduction,
            ) = utils.get_flops_and_params(self.args)
            self.logger.info(
                f"Params_baseline: {Params_baseline:.2f}M, Params: {Params:.2f}M, Params reduction: {Params_reduction:.2f}%"
            )
            self.logger.info(
                f"Flops_baseline: {Flops_baseline:.2f}M, Flops: {Flops:.2f}M, Flops reduction: {Flops_reduction:.2f}%"
            )
        except AttributeError:
            self.logger.warning("Function get_flops_and_params not found in utils. Skipping FLOPs and Params calculation.")

    def main(self):
        self.result_init()
        self.setup_seed()
        self.dataload()
        self.build_model()
        self.define_loss()
        self.define_optim()
        self.finetune()

import os
import torch
import torch.nn as nn
from tqdm import tqdm
from thop import profile
from model.teacher.ResNet import ResNet_50_hardfakevsreal
from model.student.ResNet_sparse_video import ResNet_50_sparse_uadfv
from data.video_data import create_uadfv_dataloaders  # حتماً این خط باشه


class Test:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.test_batch_size = args.test_batch_size
        self.sparsed_student_ckpt_path = args.sparsed_student_ckpt_path
        self.num_frames = getattr(args, 'num_frames', 32)

        # عدد دقیق معلم شما (از اجرای قبلی)
        self.teacher_video_flops = 170.59  # GFLOPs برای ۳۲ فریم

    def build_models(self):
        print("بارگذاری مدل معلم (فقط برای مقایسه)...")
        teacher = ResNet_50_hardfakevsreal()
        teacher.fc = nn.Linear(teacher.fc.in_features, 1)
        teacher.to(self.device)
        teacher.eval()

        print("بارگذاری مدل دانشجوی هرس‌شده...")
        student = ResNet_50_sparse_uadfv()
        student.dataset_type = "uadfv"

        ckpt = torch.load(self.sparsed_student_ckpt_path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("student", ckpt)

        if list(state_dict.keys())[0].startswith('module.'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                new_state_dict[k[7:]] = v
            state_dict = new_state_dict

        student.load_state_dict(state_dict, strict=True)
        student.to(self.device)
        student.eval()
        student.ticket = True  # پرونینگ فعال
        return teacher, student

    def test(self):
        teacher, student = self.build_models()

        dummy = torch.randn(1, 3, 256, 256).to(self.device)
        flops_frame, params = profile(student, inputs=(dummy,), verbose=False)
        flops_video = flops_frame * self.num_frames / 1e9  # GFLOPs

        print("\n" + "="*85)
        
        print("="*85)
        print(f"(Teacher)         : 170.59 GFLOPs")
        print(f" (Student)        : {self.num_frames:2d} فریم → {flops_video:7.2f} GFLOPs")
        print("-"*85)
        reduction = (self.teacher_video_flops - flops_video) / self.teacher_video_flops * 100
        print(f" FLOPs Reduction : {reduction:6.2f}%")
        print(f" Params Reduction : {(23.51 - params/1e6)/23.51*100:6.2f}%")
        print("="*85)

        # تست دقت
        _, _, test_loader = create_uadfv_dataloaders(
            root_dir=self.args.dataset_dir,
            num_frames=self.num_frames,
            image_size=256,
            train_batch_size=1,
            eval_batch_size=self.test_batch_size,
            num_workers=4,
            pin_memory=True,
            ddp=False
        )

        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in tqdm(test_loader, desc="Testing Accuracy", ncols=100):
                videos = videos.to(self.device)
                labels = labels.to(self.device).float()

                B, T, C, H, W = videos.shape
                videos = videos.view(-1, C, H, W)

                logits, _ = student(videos)
                logits = logits.view(B, T).mean(dim=1)
                preds = (torch.sigmoid(logits) > 0.5).float()

                correct += (preds == labels).sum().item()
                total += B

        accuracy = 100. * correct / total
        print(f"\nدقت نهایی روی مجموعه تست (با {self.num_frames} فریم): {accuracy:.2f}%")
        print(f"کاهش FLOPs نسبت به معلم: {reduction:.2f}% → فقط {flops_video:.2f} GFLOPs!")

    def main(self):  # این تابع رو اضافه کردم!
        self.test()


class Args:
    def __init__(self):
        self.dataset_dir = "/kaggle/input/uadfv-dataset/UADFV"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_batch_size = 8
        self.sparsed_student_ckpt_path = "/kaggle/working/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt"  # مسیر دقیق خودت
        self.num_frames = 32


if __name__ == '__main__':
    args = Args()
    tester = Test(args)
    tester.main()  # حالا کار می‌کنه!

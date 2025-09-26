import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
from model.student.ResNet_sparse import ResNet_50_sparse_rvf10k  # مدل برای rvf10k
from data.dataset import Dataset_selector  # از فایل dataset.py شما

# کلاس هوک برای استخراج activation
class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.activation = None
    def hook_fn(self, module, input, output):
        self.activation = output
    def close(self):
        self.hook.remove()

# تابع برای نمایش نقشه ویژگی‌ها
def visualize_feature_maps(activation, layer_name, num_channels_to_show=16, save_path=None):
    activation = activation.detach().cpu().squeeze(0)  # حذف batch dimension
    num_channels = activation.shape[0]
    num_channels_to_show = min(num_channels, num_channels_to_show)  # محدود کردن تعداد کانال‌ها
    
    # تنظیم تعداد ردیف‌ها و ستون‌ها
    rows = int(np.ceil(num_channels_to_show / 4))
    plt.figure(figsize=(15, 4 * rows))
    
    for i in range(num_channels_to_show):
        plt.subplot(rows, 4, i + 1)
        feature_map = activation[i].numpy()
        # نرمال‌سازی برای نمایش
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        plt.imshow(feature_map, cmap='viridis')  # می‌تونید cmap رو به 'gray' یا 'hot' تغییر بدید
        plt.title(f'{layer_name} Channel {i}')
        plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Feature maps saved as '{save_path}'")
    plt.show()

# پارامترها
checkpoint_path = '/kaggle/input/kdfs-21-mordad-10k-new-pearson-final-data/results/run_resnet50_imagenet_prune1/student_model/resnet50_sparse_best.pt'  # مسیر واقعی رو بذارید
rvf10k_train_csv = '/kaggle/input/rvf10k/train.csv'
rvf10k_valid_csv = '/kaggle/input/rvf10k/valid.csv'
rvf10k_root_dir = '/kaggle/input/rvf10k'
image_size = 256  # برای rvf10k
num_channels_to_show = 16  # تعداد کانال‌هایی که می‌خواهید نمایش داده بشن

# شبیه‌سازی args برای سازگاری با کد اصلی
class Args:
    def __init__(self):
        self.dataset_mode = 'rvf10k'
        self.gumbel_start_temperature = 5.0  # نمونه، از کد اصلی خودتون بردارید
        self.gumbel_end_temperature = 1.0    # نمونه
        self.num_epochs = 100                # نمونه
        self.dataset_type = 'rvf10k'

args = Args()

# لود دیتاست rvf10k
dataset = Dataset_selector(
    dataset_mode='rvf10k',
    rvf10k_train_csv=rvf10k_train_csv,
    rvf10k_valid_csv=rvf10k_valid_csv,
    rvf10k_root_dir=rvf10k_root_dir,
    train_batch_size=1,  # برای گرفتن یک تصویر
    eval_batch_size=1,
    num_workers=0,  # برای ساده‌سازی در visualization
    pin_memory=False,
    ddp=False  # چون اینجا DDP لازم نیست
)

# گرفتن یک تصویر از validation set
val_loader = dataset.loader_val
image, label = next(iter(val_loader))  # اولین تصویر از validation
image = image.to('cuda' if torch.cuda.is_available() else 'cpu')
label = label.item()  # برای نمایش لیبل (real یا fake)

# لود مدل
model = ResNet_50_sparse_rvf10k(
    gumbel_start_temperature=args.gumbel_start_temperature,
    gumbel_end_temperature=args.gumbel_end_temperature,
    num_epochs=args.num_epochs,
)
model.dataset_type = args.dataset_type
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# لود state_dict از چک‌پوینت
ckpt = torch.load(checkpoint_path, map_location='cpu')
state_dict = ckpt['student']
if list(state_dict.keys())[0].startswith('module.'):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
model.eval()
model.ticket = True  # برای اعمال ماسک‌ها

# انتخاب لایه‌ها
first_layer = model.layer1[0].conv1  # اولین لایه از اولین بلوک
second_layer = model.layer1[0].conv2  # دومین لایه از اولین بلوک

# نمایش تصویر ورودی
image_np = image.cpu().squeeze(0).permute(1, 2, 0).numpy()
image_np = (image_np * np.array([0.2486, 0.2238, 0.2211]) + np.array([0.5212, 0.4260, 0.3811])).clip(0, 1)  # denormalize
plt.figure(figsize=(5, 5))
plt.imshow(image_np)
plt.title(f'Input Image (Label: {"Real" if label == 1 else "Fake"})')
plt.axis('off')
plt.show()

# استخراج و نمایش feature maps برای لایه اول
hook_first = Hook(first_layer)
model(image)
vis_first = hook_first.activation
hook_first.close()
visualize_feature_maps(vis_first, 'First_Layer', num_channels_to_show, save_path='feature_maps_first_layer.png')

# استخراج و نمایش feature maps برای لایه دوم
hook_second = Hook(second_layer)
model(image)
vis_second = hook_second.activation
hook_second.close()
visualize_feature_maps(vis_second, 'Second_Layer', num_channels_to_show, save_path='feature_maps_second_layer.png')

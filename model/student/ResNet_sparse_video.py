import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from thop import profile
# فرض بر این است که فایل layer.py در کنار این فایل قرار دارد
# و شامل کلاس SoftMaskedConv2d است.
from .layer import SoftMaskedConv2d

class MaskedNet(nn.Module):
  
    def __init__(self, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs
        self.gumbel_temperature = gumbel_start_temperature
        self.ticket = False  # حالتی که در آن ماسک‌ها باینری شده‌اند
        self.mask_modules = []

    def checkpoint(self):
        """از وضعیت فعلی وزن‌ها و ماسک‌ها یک نسخه پشتیبان تهیه می‌کند."""
        for m in self.mask_modules:
            m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        """وزن‌ها را به آخرین نقطه چک‌پوینت بازمی‌گرداند."""
        for m in self.mask_modules:
            m.rewind_weights()
        for m in self.modules():
            if hasattr(m, 'checkpoint'):
                m.load_state_dict(m.checkpoint)

    def update_gumbel_temperature(self, epoch):
        """دمای گامبل را بر اساس اپوک فعلی به‌روزرسانی می‌کند."""
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        """
        FLOPs (تعداد عملیات شناور) را برای مدل بر روی یک تصویر ورودی محاسبه می‌کند.
        این محاسبه به اندازه تصویر ورودی بستگی دارد که از طریق dataset_type مشخص می‌شود.
        """
        device = next(self.parameters()).device
        Flops_total = torch.tensor(0.0, device=device)
        
        # دیکشنری اندازه تصاویر ورودی برای هر دیتاست
        image_sizes = {
            "hardfakevsrealfaces": 300,
            "rvf10k": 256,
            "140k": 256,
            "uadfv": 256  # اندازه تصویر برای دیتاست UADFV
        }
        
        dataset_type = getattr(self, "dataset_type", "hardfakevsrealfaces")
        input_size = image_sizes.get(dataset_type, 256) # مقدار پیش‌فرض 256 در صورت عدم وجود
        
        # محاسبه FLOPs برای لایه‌های اولیه (conv1, bn1)
        conv1_h = (input_size - 7 + 2 * 3) // 2 + 1
        maxpool_h = (conv1_h - 3 + 2 * 1) // 2 + 1
        
        Flops_total = Flops_total + (
            conv1_h * conv1_h * 7 * 7 * 3 * 64 +  # FLOPs for conv1
            conv1_h * conv1_h * 64                 # FLOPs for bn1 (simplified)
        )
        
        # محاسبه FLOPs برای بلوک‌های ResNet با کانولوشن‌های ماسک‌دار
        for i, m in enumerate(self.mask_modules):
            m = m.to(device)
            Flops_shortcut_conv = 0
            Flops_shortcut_bn = 0
            
            # محاسبات مخصوص ResNet-50 (48 لایه ماسک‌دار)
            if len(self.mask_modules) == 48:
                if i % 3 == 0:  # لایه اول در هر بلوک Bottleneck
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        m.in_channels * m.mask.sum()
                    )
                else:  # لایه‌های دوم و سوم در هر بلوک Bottleneck
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        self.mask_modules[i - 1].mask.to(device).sum() * m.mask.sum()
                    )
                Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
                
                # محاسبه FLOPs برای اتصال کوتاه (shortcut) در صورت وجود
                if i % 3 == 2 and m.stride != 1:
                     Flops_shortcut_conv = (
                        m.feature_map_h * m.feature_map_w * 1 * 1 *
                        (m.out_channels // 4) * m.out_channels
                    )
                     Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels

            Flops_total = (
                Flops_total + Flops_conv + Flops_bn + Flops_shortcut_conv + Flops_shortcut_bn
            )
        return Flops_total

    def get_video_flops_sampled(self, num_sampled_frames):
        flops_per_frame = self.get_flops()
        total_video_flops = flops_per_frame * num_sampled_frames
        return total_video_flops

# کلاس‌های BasicBlock_sparse و Bottleneck_sparse بدون تغییر باقی می‌مانند
# ... (کدهای این کلاس‌ها را از پاسخ قبلی کپی کنید) ...
class BasicBlock_sparse(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
    def forward(self, x, ticket):
        out = F.relu(self.bn1(self.conv1(x, ticket)))
        out = self.bn2(self.conv2(out, ticket))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class Bottleneck_sparse(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SoftMaskedConv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
    def forward(self, x, ticket):
        out = F.relu(self.bn1(self.conv1(x, ticket)))
        out = F.relu(self.bn2(self.conv2(out, ticket)))
        out = self.bn3(self.conv3(out, ticket))
        out += self.downsample(x)
        out = F.relu(out)
        return out

class ResNet_sparse(MaskedNet):
    def __init__(self, block, num_blocks, num_classes=1, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200, dataset_type="hardfakevsrealfaces"):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)
        self.in_planes = 64
        self.dataset_type = dataset_type
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.feat1 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, kernel_size=1)
        self.feat2 = nn.Conv2d(128 * block.expansion, 128 * block.expansion, kernel_size=1)
        self.feat3 = nn.Conv2d(256 * block.expansion, 256 * block.expansion, kernel_size=1)
        self.feat4 = nn.Conv2d(512 * block.expansion, 512 * block.expansion, kernel_size=1)
        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]
        self.mask_modules = [m.to(next(self.parameters()).device) for m in self.mask_modules]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        feature_list = []
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        for block in self.layer1: out = block(out, self.ticket)
        feature_list.append(self.feat1(out))
        for block in self.layer2: out = block(out, self.ticket)
        feature_list.append(self.feat2(out))
        for block in self.layer3: out = block(out, self.ticket)
        feature_list.append(self.feat3(out))
        for block in self.layer4: out = block(out, self.ticket)
        feature_list.append(self.feat4(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list

def ResNet_50_sparse_uadfv(gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
    return ResNet_sparse(block=Bottleneck_sparse, num_blocks=[3, 4, 6, 3], num_classes=1, gumbel_start_temperature=gumbel_start_temperature, gumbel_end_temperature=gumbel_end_temperature, num_epochs=num_epochs, dataset_type="uadfv")

def ResNet_50_sparse_rvf10k(gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
    return ResNet_sparse(block=Bottleneck_sparse, num_blocks=[3, 4, 6, 3], num_classes=1, gumbel_start_temperature=gumbel_start_temperature, gumbel_end_temperature=gumbel_end_temperature, num_epochs=num_epochs, dataset_type="rvf10k")

# --- مثال نحوه استفاده ---
if __name__ == '__main__':

    try:
        model = ResNet_50_sparse_uadfv()
        model.eval() # قرار دادن مدل در حالت ارزیابی

        video_duration = 11.6  # مدت زمان واقعی ویدیو به ثانیه
        video_fps = 30.00      # نرخ فریم واقعی ویدیو

        # 3. محاسبه FLOPs برای کل ویدیو با استفاده از تابع جدید
        total_video_flops = model.get_video_flops(video_duration_seconds=video_duration, fps=video_fps)
        
        print(f"--- FLOPs برای پردازش ویدیوی /kaggle/input/uadfv-dataset/UADFV/real/0008.mp4 ---")
        print(f"پارامترهای ویدیو: مدت زمان = {video_duration} ثانیه, نرخ فریم = {video_fps} FPS")
        print(f"تعداد کل فریم‌ها: {int(video_duration * video_fps)}")
        print(f"مجموع FLOPs: {total_video_flops / 1e12:.2f} TFLOPs")

    except NameError:
        print("\nخطا: کلاس SoftMaskedConv2d یافت نشد.")
        print("لطفاً مطمئن شوید که فایل layer.py به درستی import شده و کلاس مورد نظر در آن تعریف شده است.")
    except Exception as e:
        print(f"\nیک خطای پیش‌بینی نشده رخ داد: {e}")

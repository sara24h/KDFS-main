import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from thop import profile

class SoftMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels)) if bias else None
        self.init_weight()
        self.init_mask()
        self.mask = torch.ones([out_channels, 1, 1, 1])
        self.gumbel_temperature = 1.0
        self.feature_map_h = 0
        self.feature_map_w = 0

    def init_weight(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, 2, 1, 1))
        nn.init.kaiming_normal_(self.mask_weight)

    def compute_mask(self, ticket):
        if torch.isnan(self.mask_weight).any():
            self.logger.error(f"NaN in mask_weight for layer {self.layer_name}")
            raise ValueError("NaN in mask_weight")
        if torch.isinf(self.mask_weight).any():
            self.logger.error(f"Inf in mask_weight for layer {self.layer_name}")
            raise ValueError("Inf in mask_weight")
    
        mask_weight_fp32 = self.mask_weight.float() 
        if ticket:
            mask = torch.argmax(mask_weight_fp32, dim=1).float().view(-1, 1, 1, 1)
        else:
            mask = F.gumbel_softmax(
                logits=mask_weight_fp32, tau=self.gumbel_temperature, hard=True, dim=1
            )[:, 1, :, :].view(-1, 1, 1, 1)
        return mask

    def forward(self, x, ticket):
        self.mask = self.compute_mask(ticket)
        masked_weight = self.weight * self.mask
        out = F.conv2d(
            x, weight=masked_weight, bias=self.bias, stride=self.stride, padding=self.padding
        )
        self.feature_map_h, self.feature_map_w = out.shape[2], out.shape[3]
        return out

    def update_gumbel_temperature(self, temperature):
        self.gumbel_temperature = temperature

    def checkpoint(self):
        self.checkpoint_state = self.state_dict()

    def rewind_weights(self):
        if hasattr(self, 'checkpoint_state'):
            self.load_state_dict(self.checkpoint_state)

class MaskedNet(nn.Module):
    def __init__(self, gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200):
        super().__init__()
        self.gumbel_start_temperature = gumbel_start_temperature
        self.gumbel_end_temperature = gumbel_end_temperature
        self.num_epochs = num_epochs
        self.gumbel_temperature = gumbel_start_temperature
        self.ticket = False
        self.mask_modules = []

    def checkpoint(self):
        for m in self.mask_modules:
            m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules:
            m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)

    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        device = next(self.parameters()).device
        Flops_total = torch.tensor(0.0, device=device)
        image_sizes = {
            "hardfakevsrealfaces": 300,
            "rvf10k": 256,
            "140k": 256  # اضافه کردن اندازه تصویر برای دیتاست 140k
        }
        dataset_type = getattr(self, "dataset_type", "hardfakevsrealfaces")
        input_size = image_sizes.get(dataset_type, 256)  # مقدار پیش‌فرض 256 در صورت عدم وجود
        
        conv1_h = (input_size - 7 + 2 * 3) // 2 + 1
        maxpool_h = (conv1_h - 3 + 2 * 1) // 2 + 1
        conv1_w = conv1_h
        maxpool_w = maxpool_h
        
        Flops_total = Flops_total + (
            conv1_h * conv1_w * 7 * 7 * 3 * 64 +
            conv1_h * conv1_w * 64
        )
        
        for i, m in enumerate(self.mask_modules):
            m = m.to(device)
            Flops_shortcut_conv = 0
            Flops_shortcut_bn = 0
            if len(self.mask_modules) == 48:  # برای ResNet-50
                if i % 3 == 0:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        m.in_channels * m.mask.sum()
                    )
                else:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        self.mask_modules[i - 1].mask.to(device).sum() * m.mask.sum()
                    )
                Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
                if i % 3 == 2:
                    Flops_shortcut_conv = (
                        m.feature_map_h * m.feature_map_w * 1 * 1 *
                        (m.out_channels // 4) * m.out_channels
                    )
                    Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels
            elif len(self.mask_modules) in [16, 32]:  # برای مدل‌های دیگر
                if i % 2 == 0:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        m.in_channels * m.mask.sum()
                    )
                else:
                    Flops_conv = (
                        m.feature_map_h * m.feature_map_w * m.kernel_size * m.kernel_size *
                        self.mask_modules[i - 1].mask.to(device).sum() * m.mask.sum()
                    )
                Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
                if i % 2 == 1 and i != 1:
                    Flops_shortcut_conv = (
                        m.feature_map_h * m.feature_map_w * 1 * 1 *
                        m.out_channels * m.out_channels
                    )
                    Flops_shortcut_bn = m.feature_map_h * m.feature_map_w * m.out_channels

            Flops_total = (
                Flops_total + Flops_conv + Flops_bn + Flops_shortcut_conv + Flops_shortcut_bn
            )
        return Flops_total

class BasicBlock_sparse(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = SoftMaskedConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = SoftMaskedConv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
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
        self.conv2 = SoftMaskedConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = SoftMaskedConv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
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
    def __init__(
        self,
        block,
        num_blocks,
        num_classes=1,  # تغییر به 1 برای خروجی باینری
        gumbel_start_temperature=2.0,
        gumbel_end_temperature=0.5,
        num_epochs=200,
        dataset_type="hardfakevsrealfaces"
    ):
        super().__init__(
            gumbel_start_temperature,
            gumbel_end_temperature,
            num_epochs,
        )
        self.in_planes = 64
        self.dataset_type = dataset_type

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(7))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        if block == BasicBlock_sparse:
            expansion = 1
        elif block == Bottleneck_sparse:
            expansion = 4
        self.feat1 = nn.Conv2d(64 * expansion, 64 * expansion, kernel_size=1)
        self.feat2 = nn.Conv2d(128 * expansion, 128 * expansion, kernel_size=1)
        self.feat3 = nn.Conv2d(256 * expansion, 256 * expansion, kernel_size=1)
        self.feat4 = nn.Conv2d(512 * expansion, 512 * expansion, kernel_size=1)

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

        for block in self.layer1:
            out = block(out, self.ticket)
        feature_list.append(self.feat1(out))

        for block in self.layer2:
            out = block(out, self.ticket)
        feature_list.append(self.feat2(out))

        for block in self.layer3:
            out = block(out, self.ticket)
        feature_list.append(self.feat3(out))

        for block in self.layer4:
            out = block(out, self.ticket)
        feature_list.append(self.feat4(out))

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out, feature_list

def ResNet_50_sparse_hardfakevsreal(
    gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200
):
    return ResNet_sparse(
        block=Bottleneck_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=1,  # تغییر به 1 برای خروجی باینری
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
        dataset_type="hardfakevsrealfaces"
    )

def ResNet_50_sparse_rvf10k(
    gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200
):
    return ResNet_sparse(
        block=Bottleneck_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=1,  # تغییر به 1 برای خروجی باینری
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
        dataset_type="rvf10k"
    )

def ResNet_50_sparse_140k(
    gumbel_start_temperature=2.0, gumbel_end_temperature=0.5, num_epochs=200
):
    return ResNet_sparse(
        block=Bottleneck_sparse,
        num_blocks=[3, 4, 6, 3],
        num_classes=1,  # تغییر به 1 برای خروجی باینری
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
        dataset_type="140k"
    )

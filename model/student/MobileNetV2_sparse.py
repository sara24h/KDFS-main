
import torch
import torch.nn as nn
import math
from .layer import SoftMaskedConv2d
import copy

__all__ = ["mobilenetv2"]


class MaskedNet(nn.Module):
    def __init__(
        self, gumbel_start_temperature=2, gumbel_end_temperature=0.1, num_epochs=350
    ):
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
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules:
            m.rewind_weights()
        for m in self.modules():
            if (
                isinstance(m, nn.Conv2d)
                or isinstance(m, nn.BatchNorm2d)
                or isinstance(m, nn.Linear)
            ):
                m.load_state_dict(m.checkpoint)

    def update_gumbel_temperature(self, epoch):
        self.gumbel_temperature = self.gumbel_start_temperature * math.pow(
            self.gumbel_end_temperature / self.gumbel_start_temperature,
            epoch / self.num_epochs,
        )
        for m in self.mask_modules:
            m.update_gumbel_temperature(self.gumbel_temperature)

    def get_flops(self):
        Flops_total = torch.tensor(0)
        Flops_total = (
            Flops_total + 112 * 112 * 3 * 3 * 3 * 32 + 112 * 112 * 32
        )
        Flops_total = (
            Flops_total + 112 * 112 * 3 * 3 * 32 + 112 * 112 * 32
        )
        for i, m in enumerate(self.mask_modules):
            Flops_dw_conv = 0
            Flops_dw_bn = 0
            if i == 0 or (i - 1) % 2 == 0:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * m.in_channels
                    * m.mask.sum()
                )
            else:
                Flops_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * self.mask_modules[i - 1].mask.sum()
                    * m.mask.sum()
                )
            Flops_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
            if (i - 1) % 2 == 0:
                Flops_dw_conv = (
                    m.feature_map_h
                    * m.feature_map_w
                    * m.kernel_size
                    * m.kernel_size
                    * m.mask.sum()
                )
                Flops_dw_bn = m.feature_map_h * m.feature_map_w * m.mask.sum()
            Flops_total = (
                Flops_total + Flops_conv + Flops_bn + Flops_dw_conv + Flops_dw_bn
            )
        return Flops_total


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn_sparse(inp, oup):
    return nn.Sequential(
        SoftMaskedConv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual_sparse(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup
        if expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                SoftMaskedConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                SoftMaskedConv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                SoftMaskedConv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x, ticket):
        out = x
        for layer in self.conv:
            if isinstance(layer, SoftMaskedConv2d):
                out = layer(out, ticket)
            else:
                out = layer(out)
        if self.identity:
            return x + out
        else:
            return out


class MobileNetV2_sparse(MaskedNet):
    def __init__(
        self,
        num_classes=1,
        width_mult=1.0,
        gumbel_start_temperature=2,
        gumbel_end_temperature=0.1,
        num_epochs=350,
    ):
        super().__init__(gumbel_start_temperature, gumbel_end_temperature, num_epochs)
        
        teacher_feature_dims = {
            "stage1": 24, # مطابق با کانال خروجی معلم بعد از استیج مربوطه
            "stage2": 32,
            "stage3": 96,
            "stage4": 320
        }
        
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2],
            [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        
        block = InvertedResidual_sparse
        # شمارنده برای پیدا کردن ایندکس صحیح لایه‌ها
        layer_index_counter = 1
        
        for num, (t, c, n, s) in enumerate(self.cfgs):
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(
                    block(input_channel, output_channel, stride, t)
                )
                input_channel = output_channel
                
                # تعریف لایه‌های تبدیل بعد از هر استیج کلیدی
                if layer_index_counter == 3: # end of c=24 stage
                    self.convert1 = nn.Conv2d(output_channel, teacher_feature_dims["stage1"], kernel_size=1)
                elif layer_index_counter == 6: # end of c=32 stage
                    self.convert2 = nn.Conv2d(output_channel, teacher_feature_dims["stage2"], kernel_size=1)
                elif layer_index_counter == 13: # end of c=96 stage
                    self.convert3 = nn.Conv2d(output_channel, teacher_feature_dims["stage3"], kernel_size=1)
                elif layer_index_counter == 17: # end of c=320 stage
                    self.convert4 = nn.Conv2d(output_channel, teacher_feature_dims["stage4"], kernel_size=1)

                layer_index_counter += 1
                
        self.features = nn.ModuleList(layers)
        
        output_channel_final = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn_sparse(input_channel, output_channel_final)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel_final, num_classes)
        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]
        self._initialize_weights()

    def forward(self, x):
        feature_list = []
        out = x

        teacher_extraction_indices = {3, 6, 13, 17}
        convert_layer_map = {
            3: self.convert1,
            6: self.convert2,
            13: self.convert3,
            17: self.convert4,
        }

        for i, layer in enumerate(self.features):

            if i == 0:
                out = layer(out)

            else:
                out = layer(out, self.ticket)

            if i in teacher_extraction_indices:
                convert_layer = convert_layer_map[i]
                feature_list.append(convert_layer(out))

        for layer in self.conv:
            if isinstance(layer, SoftMaskedConv2d):
                out = layer(out, self.ticket)
            else:
                out = layer(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)

        return out, feature_list

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, SoftMaskedConv2d)):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels if isinstance(m, nn.Conv2d) else m.kernel_size * m.kernel_size * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def MobileNetV2_sparse_deepfake(
    gumbel_start_temperature=2, gumbel_end_temperature=0.5, num_epochs=350
):
    return MobileNetV2_sparse(
        num_classes=1,
        width_mult=1.0,
        gumbel_start_temperature=gumbel_start_temperature,
        gumbel_end_temperature=gumbel_end_temperature,
        num_epochs=num_epochs,
    )

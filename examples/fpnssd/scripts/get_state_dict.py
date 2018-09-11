import os
import torch
from torchcv.models.fpnssd import FPNMobileNetV2SSD512


# model_dir = './examples/fpnssd/model'
# params = torch.load(os.path.join(model_dir, 'resnet50-19c8e357.pth'))

# net = FPNSSD512(num_classes=9)
# net.fpn.load_state_dict(params, strict=False)
# torch.save(net.state_dict(), os.path.join(model_dir, 'fpnssd512_resnet50.pth'))

model_dir = './examples/fpnssd/model'
params = torch.load(os.path.join(model_dir, 'mobilenetv2_Top1_71.806_Top2_90.410.pth.tar'))

net = FPNMobileNetV2SSD512(num_classes=21)
net.fpn.load_state_dict(params, strict=False)
torch.save(net.state_dict(), os.path.join(model_dir, 'fpnssd512_mobilenetv2.pth'))
import torch
from  torchcv.models.ssd import SSD512
# Download PyTorch ResNet50 model from:
# https://download.pytorch.org/models/resnet50-19c8e357.pth
d = torch.load('./examples/ssd/model/ssd512_vgg16.pth')
net = SSD512(num_classes=21)
net.extractor.load_state_dict(d, strict=False)
torch.save(net.state_dict(), 'ssd512_vgg16cvt.pth')
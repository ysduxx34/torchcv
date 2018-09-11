import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.conv6 = nn.Conv2d(2048, 256, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d( 256, 256, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p8 = self.conv8(F.relu(p7))
        p9 = self.conv9(F.relu(p8))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7, p8, p9


class MobileNetV2Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(MobileNetV2Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2FPN(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    # cfg = [(1,  16, 1, 1),
    #        (6,  24, 2, 2),
    #        (6,  32, 3, 2),
    #        (6,  64, 4, 2),
    #        (6,  96, 3, 1),
    #        (6, 160, 3, 2),
    #        (6, 320, 1, 1)]

    def __init__(self):
        super(MobileNetV2FPN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(32, 1,  16, 1, 1)
        self.layer2 = self._make_layer(16, 6,  24, 2, 2)
        self.layer3 = self._make_layer(24, 6,  32, 3, 2)
        self.layer4 = self._make_layer(32, 6,  64, 4, 2)
        self.layer5 = self._make_layer(64, 6,  96, 3, 1)
        self.layer6 = self._make_layer(96, 6, 160, 3, 2)
        self.layer7 = self._make_layer(160, 6, 320, 1, 1)

        self.conv6 = nn.Conv2d( 320, 64, kernel_size=3, stride=2, padding=1)
        self.conv7 = nn.Conv2d( 64, 64, kernel_size=3, stride=2, padding=1)
        self.conv8 = nn.Conv2d( 64, 64, kernel_size=3, stride=2, padding=1)
        self.conv9 = nn.Conv2d( 64, 64, kernel_size=3, stride=2, padding=1)

        # Top-down layers
        self.toplayer = nn.Conv2d(320, 64, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)
        self.latlayer3 = nn.Conv2d(24, 64, kernel_size=1, stride=1, padding=0)

        # Smooth layers
        self.smooth1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, in_planes, expansion, out_planes, num_blocks, stride):
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
        for stride in strides:
            layers.append(MobileNetV2Block(in_planes, out_planes, expansion, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.

        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.

        Returns:
          (Variable) added feature map.

        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.

        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]

        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='bilinear', align_corners=False) + y

    def forward(self, x):
        c1 = F.relu(self.bn1(self.conv1(x)))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        c6 = self.layer5(c5)
        c7 = self.layer6(c6)
        c8 = self.layer7(c7)

        p9 = self.conv6(c8)
        p10 = self.conv7(F.relu(p9))
        p11 = self.conv8(F.relu(p10))
        p12 = self.conv9(F.relu(p11))
        # Top-down
        p8 = self.toplayer(c8)
        p5 = self._upsample_add(p8, self.latlayer1(c5))
        p4 = self._upsample_add(p5, self.latlayer2(c4))
        # TODO: add p3 to box and cls
        p3 = self._upsample_add(p8, self.latlayer3(c3))

        p5 = self.smooth1(p5)
        p4 = self.smooth2(p4)
        return p4, p5, p8, p9, p10, p11, p12

        # out = F.relu(self.bn2(self.conv2(out)))
        # # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        # out = F.avg_pool2d(out, 4)
        # out = out.view(out.size(0), -1)
        # out = self.linear(out)
        # return out

def FPN_MobileNetV2():
    return MobileNetV2FPN()

def FPN50():
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    return FPN(Bottleneck, [3,4,23,3])

def FPN152():
    return FPN(Bottleneck, [3,8,36,3])


def test():
    net = FPN50()
    fms = net(torch.randn(1,3,512,512))
    for fm in fms:
        print(fm.size())

# test()

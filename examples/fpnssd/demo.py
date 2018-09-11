import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
from torchcv.models.fpnssd import FPNMobileNetV2SSD512, FPNSSDBoxCoder
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

print('Loading model..')
net = FPNMobileNetV2SSD512(num_classes=21).to('cuda')
net = torch.nn.DataParallel(net)
# cudnn.benchmark = True
# net.load_state_dict(torch.load('/home/ysdu/torchcv/examples/fpnssd/checkpoint/ckpt.pth'))
checkpoint = torch.load('/home/ysdu/torchcv/examples/fpnssd/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_loss = checkpoint['loss']
start_epoch = checkpoint['epoch']
# net = torch.nn.DataParallel(net)
net.eval()

print('Loading image..')
img = Image.open('/home/ysdu/hardwareDisk/ysduDir/voc/VOCdevkit_test/VOC2012/JPEGImages/2008_002819.jpg')
ow = oh = 512
img = img.resize((ow,oh))

print('Predicting..')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
x = transform(img)
loc_preds, cls_preds = net(x.unsqueeze(0))

print('Decoding..')
box_coder = FPNSSDBoxCoder()
loc_preds = loc_preds.squeeze().cpu()
cls_preds = F.softmax(cls_preds.squeeze(), dim=1).cpu()
boxes, labels, scores = box_coder.decode(loc_preds, cls_preds)

print (labels)
draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()

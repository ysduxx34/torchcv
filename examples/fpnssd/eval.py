import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from torchcv.transforms import resize
from torchcv.datasets import ListDataset
from torchcv.evaluations.voc_eval import voc_eval
# from torchcv.models.ssd import FPNMobileNetV2SSD512,SSD300, SSDBoxCoder
from torchcv.models.fpnssd import FPNMobileNetV2SSD512, FPNSSDBoxCoder

from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

print('Loading model..')
# net = SSD300(num_classes=21)
# net.load_state_dict(torch.load('./examples/ssd/checkpoint/ckpt.pth'))
# net.cuda()
# net.eval()

net = FPNMobileNetV2SSD512(num_classes=21).to('cuda')
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

checkpoint = torch.load('/home/ysdu/torchcv/examples/fpnssd/checkpoint/ckpt.pth')
net.load_state_dict(checkpoint['net'])
best_loss = checkpoint['loss']
start_epoch = checkpoint['epoch']
net.cuda()
net.eval()
    
print('Preparing dataset..')
img_size = 512
def transform(img, boxes, labels):
    img, boxes = resize(img, boxes, size=(img_size,img_size))
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    return img, boxes, labels

dataset = ListDataset(root='/home/ysdu/hardwareDisk/ysduDir/voc/voc_all_images', \
                      list_file='torchcv/datasets/voc/voc07_test.txt',
                      transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
box_coder = FPNSSDBoxCoder()

pred_boxes = []
pred_labels = []
pred_scores = []
gt_boxes = []
gt_labels = []

with open('torchcv/datasets/voc/voc07_test_difficult.txt') as f:
    gt_difficults = []
    for line in f.readlines():
        line = line.strip().split()
        d = [int(x) for x in line[1:]]
        gt_difficults.append(d)

def eval(net, dataset):
    for i, (inputs, box_targets, label_targets) in enumerate(dataloader):
        print('%d/%d' % (i, len(dataloader)))
        gt_boxes.append(box_targets.squeeze(0))
        gt_labels.append(label_targets.squeeze(0))

        loc_preds, cls_preds = net(Variable(inputs.cuda(), volatile=True))
        box_preds, label_preds, score_preds = box_coder.decode(
            loc_preds.cuda().data.squeeze(),
            F.softmax(cls_preds.squeeze(), dim=1).cuda().data,
            score_thresh=0.01)

        pred_boxes.append(box_preds)
        pred_labels.append(label_preds)
        pred_scores.append(score_preds)

    print (voc_eval(
            pred_boxes, pred_labels, pred_scores,
            gt_boxes, gt_labels, gt_difficults,
            iou_thresh=0.5, use_07_metric=True))

eval(net, dataset)

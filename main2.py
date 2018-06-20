import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torchvision.transforms import cvtransforms
from lib.Siamese import ImageSiamese
from lib.siamese_resnet import res50_sia
import numpy as np
import cv2
import gc
import sys
import json

def cv2_loader(path):
    return cv2.imread(path)

def main():

    model1 = res50_sia(pretrained=True)
    del model1.base.fc

    model2 = res50_sia(pretrained=True)
    del model2.base.fc

    if args.gpu_ids == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model1 = nn.DataParallel(model1,device_ids=range(torch.cuda.device_count())).cuda()
        model1.to(device)
        model2 = nn.DataParallel(model2,device_ids=range(torch.cuda.device_count())).cuda()
        model2.to(device)
    else:
        torch.cuda.set_device(args.gpu_ids)
        model1 = nn.DataParallel(model1,device_ids=[ args.gpu_ids ]).cuda()
        model2 = nn.DataParallel(model2,device_ids=[ args.gpu_ids ]).cuda()

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            model1.load_state_dict(torch.load(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.resume_p:
        if os.path.isfile(args.resume_p):
            print("=> loading checkpoint '{}'".format(args.resume_p))
            model2.load_state_dict(torch.load(args.resume_p))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.train:
        tic = time.time()
        pair_json = args.pair_json
        data_json = args.data_json
        image_json = args.image_json
        toc = time.time()
        print('finish load data!!!   cost: {}'.format(toc-tic))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        if args.rot:
            train_dataset = ImageSiamese(
                pair_json,
                data_json,
                image_json,
                transforms.Compose([
                    cvtransforms.RGB(),
                    cvtransforms.Rot90(),
                    cvtransforms.FixResize(224),
                    transforms.ToTensor(),
                    normalize,
                ]), loader=cv2_loader)
        else:
            train_dataset = ImageSiamese(
                pair_json,
                data_json,
                image_json,
                transforms.Compose([
                    cvtransforms.RGB(),
                    cvtransforms.FixResize(224),
                    transforms.ToTensor(),
                    normalize,
                ]), loader=cv2_loader)       

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, sampler=None)

        param_groups1 = model1.parameters()
        param_groups2 = model2.parameters()
        optimizer1 = torch.optim.SGD(param_groups1, lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False)  
        optimizer2 = torch.optim.SGD(param_groups2, lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=False)  

        print('num of batches in each epoch: {}'.format(len(train_loader)))
        print('begin training!!!')

        if args.gpu_ids == -1:
            train(train_loader, model1, model2, optimizer1, optimizer2, device)
        else:
            train(train_loader, model1, model2, optimizer1, optimizer2)
    else:
        print('begin testing!!')
        test(model1, model2)

def train(train_loader, model1, model2, optimizer1, optimizer2, device= None):

    model1.train()
    model2.train()    

    for ep in range(args.epoch):
        adjust_learning_rate(optimizer1, ep)
        adjust_learning_rate(optimizer2, ep)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        end = time.time()
        for i, (batch_a, batch_b, batch_label) in enumerate(train_loader):

            if device:
                batch_a.to(device)
                batch_b.to(device)
                batch_label.to(device)

            data_time.update(time.time() - end)

            batch_label = batch_label.long()
            batch_label = Variable(batch_label.float()).cuda()
            batch_a = Variable(batch_a).cuda()
            batch_b = Variable(batch_b).cuda()

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            features_1 = model1(batch_a)
            features_2 = model2(batch_b)

            euclidean_distance = F.pairwise_distance(features_1, features_2)
            loss_contrastive = torch.mean(batch_label * torch.pow(euclidean_distance, 2) +
                (1 - batch_label) * torch.pow(torch.clamp(1.0 - euclidean_distance, min=0.0), 2))/2.0
            # loss_contrastive = torch.mean(torch.pow(euclidean_distance, 2) +
            #     torch.pow(torch.clamp(1.0 - euclidean_distance, min=0.0), 2))/2.0

            losses.update(loss_contrastive.item(), batch_a.size(0))

            loss_contrastive.backward()
            optimizer1.step()
            optimizer2.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                sys.stderr.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                   ep, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

            if i%10000 == 0:
                gc.collect()

        torch.save(model1.state_dict(), '/home/liuhuawei/yanjia/siamese/models/{}_epoch_{}.pth'.format(args.name, ep))
        torch.save(model2.state_dict(), '/home/liuhuawei/yanjia/siamese/models/{}_epoch_{}_p.pth'.format(args.name, ep))


    print('Training Done!!')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 2))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning_rate: {}'.format(lr))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def test(model1, model2):

    objectid_to_metadata = json.load(open('{}'.format(args.data_json), 'r'))
    with open('{}.txt'.format(args.test_txt_pre)) as f:
        data1 = [i.strip().split() for i in f]
    with open('{}_p.txt'.format(args.test_txt_pre)) as f:
        data2 = [i.strip().split() for i in f]

    model1 = model1.eval()
    model2 = model2.eval()

    blen = 0
    ind = 0
    for i in range(len(data1)):
        if data1[i][2] == '60':
            blen = i
            break
    for i in range(len(data1)):
        if data1[i][2] == '90':
            ind = i 
            break
    print blen

    labels = []
    distances = []
    right_count = []
    wrong_pair = []
    obj_ids1 = []
    obj_ids2 = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    for j in range(blen):
        i = j+ind
        object_id1, label, l = data1[i]
        object_id2 = data2[i][0]

        if label == 'none':
            if labels and 1 in labels:
                print '-------------------------', labels.index(1), distances.index(min(distances))
                right_count.append(int(labels.index(1)==distances.index(min(distances))))
                print 1.0 * sum(right_count) / len(right_count)

                if labels.index(1) != distances.index(min(distances)):
                    wrong_pair.append([obj_ids1[0],obj_ids2[0],obj_ids2[distances.index(min(distances))], distances[0], distances[distances.index(min(distances))]])

                    print wrong_pair

                obj_ids1 = []
                obj_ids2 = []

            labels = []
            distances = []
            continue

        object1 = objectid_to_metadata[object_id1]
        object2 = objectid_to_metadata[object_id2]

        im_a = crop_image(object1, int(l))
        im_b = crop_image(object2, int(l))

        if args.rot:
            trans = transforms.Compose([
                    cvtransforms.RGB(),
                    cvtransforms.Rot90(),
                    cvtransforms.FixResize(224),
                    transforms.ToTensor(),
                    normalize,
                    ])
        else:
            trans = transforms.Compose([
                    cvtransforms.RGB(),
                    cvtransforms.FixResize(224),
                    transforms.ToTensor(),
                    normalize,
                    ])

        batch_im_a = trans(im_a).unsqueeze(0)
        batch_im_b = trans(im_b).unsqueeze(0)

        batch_im_a = Variable(batch_im_a).cuda()            
        batch_im_b = Variable(batch_im_b).cuda()    

        features_1 = model1(batch_im_a)
        features_2 = model2(batch_im_b)

        dist = F.pairwise_distance(features_1, features_2).cpu()
        dist = dist.data.numpy()[0]

        distances.append(dist)
        print object_id1, object_id2, label, dist
        obj_ids1.append(object_id1)
        obj_ids2.append(object_id2)
        labels.append(int(label))

def crop_image(item, length):
    path = item['img_path']
    bbox = [int(item['xmin']), int(item['ymin']), int(item['xmin'])+int(item['width']), int(item['ymin'])+int(item['height'])]
    img = cv2.imread(path)
    h, w = img.shape[0:2]
    bbox[0] = max(0, bbox[0]-length)
    bbox[1] = max(0, bbox[1]-length)
    bbox[2] = min(w, bbox[2]+length)
    bbox[3] = min(h, bbox[3]+length)
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]


if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser(description="Train Siamese")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--gpu_ids', type=int, default=-1)
    parser.add_argument('--epoch', type=int, default=3)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--resume_p', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--pair_json', type=str, default=None)
    parser.add_argument('--data_json', type=str, default=None)
    parser.add_argument('--image_json', type=str, default=None)
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--test_txt_pre', type=str,default=None)
    parser.add_argument('--rot',action='store_true')

    args = parser.parse_args()
    main()


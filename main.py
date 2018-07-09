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
import MySQLdb

np.random.seed(50)

def cv2_loader(path):
    return cv2.imread(path)


def mask2bbox(stri):
    stri = stri[1:-1].split(',')
    segm = [int(i.split('.')[0]) for i in stri]
    x = segm[0::2]
    y = segm[1::2]
    assert len(x) == len(y)
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    return [xmin,ymin,xmax,ymax]


def main():

    ### create two same model 
    model1 = res50_sia(pretrained=True, fc2_relu=args.fc2_relu, fc_init=args.fc_init)
    del model1.base.fc
    model2 = res50_sia(pretrained=True, fc2_relu=args.fc2_relu, fc_init=args.fc_init)
    del model2.base.fc

    p1 = model1.state_dict()
    p2 = model2.state_dict()
    p2['base.ip1.weight'] = p1['base.ip1.weight']
    p2['base.ip2.weight'] = p1['base.ip2.weight']
    p2['base.feat.weight'] = p1['base.feat.weight']
    model2.load_state_dict(p2)

    ### set device id:
    ### default None: use all available GPU
    if args.gpu_ids == None:
        device_ids = range(torch.cuda.device_count())
    else:
        device_ids = [int(i) for i in args.gpu_ids.split(',')]

    torch.cuda.set_device(device_ids[0])
    model1 = nn.DataParallel(model1,device_ids=device_ids).cuda()
    model2 = nn.DataParallel(model2,device_ids=device_ids).cuda()

    ### load model and model_p for test or pretrain
    if args.resume:
        resume = args.resume
        resume_p = args.resume.split('.')[0] + '_p.pth'
        if os.path.isfile(resume) and os.path.isfile(resume_p):
            print("=> loading model '{}'".format(resume))
            print("=> loading model_p '{}'".format(resume_p))
            model1.load_state_dict(torch.load(resume))
            model2.load_state_dict(torch.load(resume_p))
        else:
            print("=> no model found at '{}' or no model_p found at '{}' ".format(resume, resume_p))

    ### True: train mode
    ### False: test mode 
    if args.train:
        tic = time.time()
        pair_json = args.pair_json
        data_json = args.data_json
        image_json = args.image_json
        toc = time.time()
        print('finish load data!!!   cost: {}'.format(toc-tic))

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        train_dataset = ImageSiamese(
            pair_json,
            data_json,
            image_json,
            transforms.Compose([
                cvtransforms.RGB(),
                cvtransforms.Rot90(args.rot),
                cvtransforms.FixResize(224),
                transforms.ToTensor(),
                normalize,
            ]), loader=cv2_loader, crop_method=args.crop_method)    

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

        train(train_loader, model1, model2, optimizer1, optimizer2)
    else:
        print('begin testing!!')
        ### True: new version test
        ### False: old version test
        if args.new_test:
            new_test(model1, model2)
        else:
            test(model1, model2)

def train(train_loader, model1, model2, optimizer1, optimizer2):   
    ### whether fix BN layer
    if args.fix_bn:
        model1.eval()
        model2.eval()
    else:
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

            data_time.update(time.time() - end)

            batch_label = batch_label.float().cuda(non_blocking=True)

            features_1 = model1(batch_a)
            features_2 = model2(batch_b)

            euclidean_distance = F.pairwise_distance(features_1, features_2)
            loss_contrastive = torch.mean(batch_label * torch.pow(euclidean_distance, 2) +
                (1 - batch_label) * torch.pow(torch.clamp(1.0 - euclidean_distance, min=0.0), 2))/2.0

            losses.update(loss_contrastive.item(), batch_a.size(0))

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss_contrastive.backward()
            optimizer1.step()
            optimizer2.step()

            batch_time.update(time.time() - end)
            end = time.time()

            ### print info
            if i % args.print_freq == 0:
                sys.stderr.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\n'.format(
                   ep, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

            ### data collection
            if i%10000 == 0:
                gc.collect()

        torch.save(model1.state_dict(), 'models/{}_epoch_{}.pth'.format(args.name, ep))
        torch.save(model2.state_dict(), 'models/{}_epoch_{}_p.pth'.format(args.name, ep))


    print('Training Done!!')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 2))
    ep1, ep2 = int(args.decay_epoch.split(',')[0]),int(args.decay_epoch.split(',')[1]) 
    if epoch >= ep1 and epoch < ep2:
        lr = args.lr * 0.1
    elif epoch >= ep2:
        lr = args.lr * 0.01
    else:
        lr = args.lr
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
    """old version test"""
    objectid_to_metadata = json.load(open('{}'.format(args.data_json), 'r'))
    with open('{}.txt'.format(args.test_txt_pre)) as f:
        data1 = [i.strip().split() for i in f]
    with open('{}_p.txt'.format(args.test_txt_pre)) as f:
        data2 = [i.strip().split() for i in f]

    model1 = model1.eval()
    model2 = model2.eval()

    labels = []
    distances = []
    right_count = []
    wrong_pair = []
    obj_ids1 = []
    obj_ids2 = []

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    for i in range(len(data1)):
        object_id1, label = data1[i]
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

        im_a = crop_image(object1, args.crop)
        im_b = crop_image(object2, args.crop)

        trans = transforms.Compose([
                cvtransforms.RGB(),
                cvtransforms.Rot90(args.rot),
                cvtransforms.FixResize(224),
                transforms.ToTensor(),
                normalize,
                ])

        batch_im_a = trans(im_a).unsqueeze(0)
        batch_im_b = trans(im_b).unsqueeze(0)

        batch_im_a = batch_im_a.cuda()            
        batch_im_b = batch_im_b.cuda()    

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


def data_prepare(img_list):
    """ generate annotations data
    Args:
        img_list (list): pairs of paired image id. \
            Example:  [[125123, 125124], [...], [...], ...]
    Returns:
        annotations (dict): key is img_id, value is a list [{}, {}, {} ,...]. \
            In this list each dict contains 'bbox', 'type', 'center', 'w', 'h', 'img_id', \
            'pair_id', 'location', 'path', 'object_id', 'url'. \
            In these keys we emphasize 'pair_id' and 'location' which can not be easily figured out, \
            'pair_id' and 'location' are related to 'path', for instance, 'path' is '/core1/data/image/citybox_2/image_00000073_0.jpg', \
            and then 'pair_id' is 73 and 'location' is 0. \
            Since the name of image we save is in pattern 'image_000000{0}_{1}.jpg', we use {0} to denote a pair of image. If two images are in pair, \
            then their {0} should be the same, and {1} denotes this pic photoed by left or right cam. \
            Example: {125123: [{'bbox':, ...}, {...}, ...], 125124: [{...},{...},...], ... }
        loc (list): list of all location
            Example: [0,1,2,3]
    """
    src = args.src.split(',')
    src_format = ''''''
    for i in range(len(src)):
        src_format += ''''{0[%d]}','''%i
    src_format = src_format[:-1]

    img_list_format = ''
    for i in range(len(img_list)):
        img_list_format += '{1[%d]},'%i
    img_list_format = img_list_format[:-1]

    conn = MySQLdb.connect(host=args.host, user=args.user, passwd=args.passwd, db=args.db)
    cur = conn.cursor()

    if args.rot != 0:
        cur.execute(('''select o.points_pixel, o.img_id, o.type, i.url, o.id, concat('http://', i.path) from object_point o inner join image i on o.img_id = i.id \
            where o.src_src in (%s) and i.id in (%s) \
            and o.status=0 and i.point_status = 4 and i.verify_status = 1'''%(src_format, img_list_format)).format(src, img_list))
        data = cur.fetchall()

        annotations = {}
        loc = []
        for d in data:
            ### get rid of obj type of which is 728 (blurry)
            if int(d[2]) == 728:
                continue

            x0, y0, x1, y1 = mask2bbox(d[0])
            pair_id, location = int(d[3].split('_')[-2]), int(d[3].split('_')[-1].split('.')[0])

            if location not in loc:
                loc.append(location)

            if d[1] in annotations:
                annotations[d[1]].append({'bbox': [x0,y0,x1,y1], 'type': int(d[2]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': x1-x0, 'h': y1-y0, 'img_id': int(d[1]), 'pair_id': pair_id, 'location': location, 'path': d[3], 'object_id': int(d[4]), 'url': d[5]})
            else:
                annotations[d[1]] = [{'bbox': [x0,y0,x1,y1], 'type': int(d[2]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': x1-x0, 'h': y1-y0, 'img_id': int(d[1]), 'pair_id': pair_id, 'location': location, 'path': d[3], 'object_id': int(d[4]), 'url': d[5]}]

    else:
        cur.execute(('''
            select o.x_pixel, o.y_pixel, o.width_pixel, o.height_pixel, o.img_id, o.type, i.url, o.id from object o inner join image i on o.img_id = i.id \
            where o.src_src in (%s) and i.id in (%s) \
            and o.box_status = 0 and i.verify_status = 1'''%(src_format, img_list_format)).format(src, img_list))
        data = cur.fetchall()

        annotations = {}
        loc = []
        for d in data:
            ### get rid of obj type of which is 728 (blurry)
            if int(d[6]) == 728:
                continue

            x0, y0, x1, y1 = int(d[0]), int(d[1]), int(d[0] + d[2]), int(d[1] + d[3])
            pair_id, location = int(d[6].split('_')[-2]), int(d[6].split('_')[-1].split('.')[0])

            if location not in loc:
                loc.append(location)

            if d[4] in annotations:
                annotations[d[4]].append({'bbox': [x0,y0,x1,y1], 'type': int(d[5]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': int(d[2]), 'h': int(d[3]), 'img_id': int(d[4]), 'pair_id': pair_id, 'location': location, 'path': d[6], 'object_id': int(d[7])})
            else:
                annotations[d[4]] = [{'bbox': [x0,y0,x1,y1], 'type': int(d[5]), 'center': [(x0+x1)/2, (y0+y1)/2], 'w': int(d[2]), 'h': int(d[3]), 'img_id': int(d[4]), 'pair_id': pair_id, 'location': location, 'path': d[6], 'object_id': int(d[7])}]

    return annotations, loc

def new_test(model1, model2):
    '''new verstion test'''
    global trans
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    trans = transforms.Compose([
            cvtransforms.RGB(),
            cvtransforms.Rot90(args.rot),
            cvtransforms.FixResize(224),
            transforms.ToTensor(),
            normalize,
            ])

    model1 = model1.eval()
    model2 = model2.eval()

    with open('{}.txt'.format(args.test_txt_pre)) as f:
        data1 = [i.strip().split() for i in f]
    with open('{}_p.txt'.format(args.test_txt_pre)) as f:
        data2 = [i.strip().split() for i in f]

    obj1_to_obj2 = {}
    for i in range(len(data1)):
        if data1[i][0] == 'none':
            continue
        object_id1 = int(data1[i][0])
        object_id2 = int(data2[i][0])

        if object_id1 in obj1_to_obj2.keys():
            obj1_to_obj2[object_id1] += [object_id2]
        else:
            obj1_to_obj2[object_id1] = [object_id2]

    f = open('{}'.format(args.test), 'r')
    lines = f.readlines()
    test_image_id_list = []
    for i in lines:
        test_image_id_list.extend([int(i.split(',')[0]), int(i.split(',')[1])])

    data, loc = data_prepare(test_image_id_list)
    num_pair = (max(loc)+1)//2
    pair_loc = []
    for i in loc:
        if i//2 not in pair_loc:
            pair_loc.append(i//2)
    print pair_loc

    pair_annotations = []
    for i in range(len(data)//2):
        img_id1 = test_image_id_list[i*2]
        img_id2 = test_image_id_list[i*2+1]
        pair_annotations.append([data[img_id1], data[img_id2]])

    result = []
    wrong_pair = []

    for annotations1, annotations2 in pair_annotations:
        img_a = cv2.imread(annotations1[0]['path'])
        img_b = cv2.imread(annotations2[0]['path'])
        h, w = img_a.shape[0:2]

        annotations1 = update_anno_info(annotations1, h, w)
        annotations2 = update_anno_info(annotations2, h, w)

        inter_annotations1 = [i for i in annotations1 if i['w'] > 0 and i['h'] >0]
        inter_annotations2 = [i for i in annotations2 if i['w'] > 0 and i['h'] >0]

        inter_annotations1 = update_anno_siamese_feature(inter_annotations1, model1, img_a, h, w)
        inter_annotations2 = update_anno_siamese_feature(inter_annotations2, model2, img_b, h, w)

        len1, len2 = len(inter_annotations1), len(inter_annotations2)
        dist_map = np.zeros((len1, len2))

        for i in range(len1):
            for j in range(len2):
                anno1, anno2 = inter_annotations1[i], inter_annotations2[j]
                dist_map[i,j] = np.linalg.norm(anno1['siamese_feature'] - anno2['siamese_feature'])

        for i in range(len1):
            obj_id1 = inter_annotations1[i]['object_id']
            if obj_id1 not in obj1_to_obj2.keys():
                continue

            obj_id2_list = obj1_to_obj2[obj_id1]
            same_obj_id2 = obj_id2_list[0]

            obj_id2_idx = [0] * len(obj_id2_list)
            for idx, anno in enumerate(inter_annotations2):
                if anno['object_id'] in obj_id2_list:
                    obj_id2_idx[obj_id2_list.index(anno['object_id'])] = idx

            dist_map_tmp = [dist_map[i,:][j] for j in obj_id2_idx]
            obj_id2 = obj_id2_list[np.argmin(dist_map_tmp)]

            if obj_id2 == same_obj_id2:
                result.append(1)
            else:
                result.append(0)
                wrong_pair.append([obj_id1, same_obj_id2, obj_id2, round(dist_map_tmp[0], 3), round(min(dist_map_tmp),3)])

        print 'right_count: ', np.sum(result), 'total_count: ', len(result), 'acc: ', 1.0 * sum(result) / len(result)

    print wrong_pair

def produce_bbox(anno, length, h, w):

    bbox = anno['bbox']
    if args.crop_method != 'fix':
        cen = anno['center']
        s_cen = np.array([w/2,h/2])
        length = length - np.linalg.norm(cen-s_cen)*60/540

    scaled_bbox = [0,0,0,0]
    scaled_bbox[0] = max(bbox[0]-length, 0)
    scaled_bbox[1] = max(bbox[1]-length, 0)
    scaled_bbox[2] = min(w, bbox[2]+length)
    scaled_bbox[3] = min(h, bbox[3]+length)
    return [int(i) for i in scaled_bbox]  

def update_anno_info(annotations, h, w):
    for anno in annotations:
        x0,y0,x1,y1 = anno['bbox']
        if x0 < 0:
            x0 = 0
        elif x1 > w:
            x1 = w
        elif y0 < 0:
            y0 = 0
        elif y1 > h:
            y1 = h
        anno['bbox'] = x0,y0,x1,y1
        anno['center'] = [(x0+x1)/2, (y0+y1)/2]
        anno['w'] = x1 - x0
        anno['h'] = y1 - y0
        anno['prob'] = anno.get('prob', 1.0)
    return annotations

def update_anno_siamese_feature(annotations, model, img, h, w):
    anno_scaled_images = []
    for anno in annotations:
        anno_scaled_bbox = produce_bbox(anno, args.crop, h, w)
        x1, y1, x2, y2 = anno_scaled_bbox
        anno_scaled_image = trans(img[y1: y2, x1: x2])
        anno_scaled_images.append(anno_scaled_image)
    anno_scaled_images_feature = get_siamese_feature(model, anno_scaled_images, args.batch_size)
    for i in range(len(annotations)):
        annotations[i]['siamese_feature'] = anno_scaled_images_feature[i]
    return annotations

def get_siamese_feature(model, images, batch_size=8):
    features = []
    st = time.time()
    while len(images):
        if len(images) > batch_size:
            imgs = [images.pop(0) for i in range(batch_size)]
            batch_im = torch.stack(imgs, 0)
            batch_im = batch_im.cuda()
            output = model(batch_im)
            output = output.data.cpu().numpy()
            features.extend([output[i] for i in range(output.shape[0])])
        else:
            imgs = [images.pop(0) for i in range(len(images))]
            batch_im = torch.stack(imgs, 0)
            batch_im = batch_im.cuda()
            output = model(batch_im)
            output = output.data.cpu().numpy()
            features.extend([output[i] for i in range(output.shape[0])])
    return features

if __name__ == '__main__':
    global args

    parser = argparse.ArgumentParser(description="Train Siamese")
    parser.add_argument('--host', type=str, default='192.168.1.12',
                        help='host of database')
    parser.add_argument('--user', type=str, default='root',
                        help='user of database')
    parser.add_argument('--passwd', type=str, default='reload123',
                        help='password of database')  
    parser.add_argument('--db', type=str, default='shampoo',
                        help='name of database')      
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--gpu_ids', type=str, default=None,
                        help='gpu ids for training and testing')
    parser.add_argument('--epoch', type=int, default=6,
                        help='num of epoches for training')
    parser.add_argument('--train', action='store_true',
                        help='whether train or not')
    parser.add_argument('--workers', type=int, default=8,
                        help='num of workers for data preprocessing')
    parser.add_argument('--name', type=str, default='',
                        help='name of trained models')
    parser.add_argument('--resume', type=str, default='',
                        help='path of model for pretraining or testing')
    parser.add_argument('--resume_p', type=str, default='',
                        help='path of model_p for pretraining or testing')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch_size for training or testing')
    parser.add_argument('--pair_json', type=str, default=None,
                        help='path of pair train json')
    parser.add_argument('--data_json', type=str, default=None,
                        help='path of metadata json')
    parser.add_argument('--image_json', type=str, default=None,
                        help='path of image json')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--test_txt_pre', type=str,default=None,
                        help='prefix of path of test txt')
    parser.add_argument('--rot', type=int, default=0,
                        help='time of rotating 90 degree')
    parser.add_argument('--decay_epoch', type=str, default='2,4',
                        help='learning rate decay epoch')
    parser.add_argument('--fix_bn', action='store_true',
                        help='whether fix BN while training')
    parser.add_argument('--test_image_txt', dest='test', type=str,default=None,
                        help='path of test image txt')
    parser.add_argument('--src_src', dest='src', type=str, default=None,
                        help='src_src of test data in mysql database')
    parser.add_argument('--crop_size', dest='crop', type=int, default=90,
                        help='length of crop')
    parser.add_argument('--new_test', action='store_true',
                        help='whether use new version test')
    parser.add_argument('--crop_method', type=str, default='fix')
    parser.add_argument('--fc2_relu', action='store_false',
                        help='if true, turn on fc2 relu layer')
    parser.add_argument('--fc_init', type=str, default='default',
                        help='initial way of fc layer')

    args = parser.parse_args()
    main()


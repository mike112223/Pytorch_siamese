import argparse
import os
import time
import json
import cv2
import sys
import numpy as np
# sys.path.insert(0, '/home/zhuyanjia/py-faster-rcnn/caffe-fast-rcnn/python')
# sys.path.insert(0, '/home/zhuyanjia/py-faster-rcnn/lib')

def cv2_loader(path):
    return cv2.imread(path)

def crop_image1(item, length):
    path = item['img_path']
    bbox = [int(item['xmin']), int(item['ymin']), int(item['xmin'])+int(item['width']), int(item['ymin'])+int(item['height'])]
    img = cv2.imread(path)
    img = img.astype(np.float32, copy=False)
    h, w = img.shape[0:2]
    bbox[0] = max(0, bbox[0]-length)
    bbox[1] = max(0, bbox[1]-length)
    bbox[2] = min(w, bbox[2]+length)
    bbox[3] = min(h, bbox[3]+length)
    return img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

def main(args):
    import caffe

    a = '/home/zhuyanjia/py-faster-rcnn/siamese_shelf/res50/merge_models/ResNet_50_deploy.prototxt'
    b = '/core1/data/home/yanjia/models/caffe_siamese_model/resnet50_citybox_orig_v17_iter_118000.caffemodel'
    net = caffe.Net(a,b,caffe.TEST)

    pair_json = '../pretrain/pair_train.json'
    data_json = '/data-4t/home/yanjia/siamese_shelf/length_data/json_data/objectid_to_metadata_all_orig.json'

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
    sys.path.insert(0, '/home/zhuyanjia/pytorch_examples/siamese/lib')

    from torchvision.transforms import cvtransforms
    from Siamese import ImageSiamese
    from siamese_resnet import res50_sia

    torch.cuda.set_device(args.gpu_ids)
    normalize = cvtransforms.SubMean(mean=np.array([[[104.0, 117.0, 123.0]]]))

    model = res50_sia(pretrained=False,siamese=True)
    del model.base.fc
    model = nn.DataParallel(model,device_ids=[ args.gpu_ids ]).cuda()

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    model.eval().cuda()


    pair = json.load(open(pair_json,'r'))
    objectid_to_metadata = json.load(open(data_json,'r'))

    obj_id1, obj_id2, l = '239892', '239924', '90'
    label = 1

    object1 = objectid_to_metadata[obj_id1]
    object2 = objectid_to_metadata[obj_id2]

    im_a = np.rot90(crop_image1(object1, int(l))).copy()
    im_b = np.rot90(crop_image1(object2, int(l))).copy()

    target_size = 224
    im_a = cv2.resize(im_a, (target_size, target_size),
        interpolation=cv2.INTER_LINEAR)
    im_b = cv2.resize(im_b, (target_size, target_size),
        interpolation=cv2.INTER_LINEAR)

    pixel_means = np.array([[[104.0, 117.0, 123.0]]])
    im_a -= pixel_means
    im_b -= pixel_means

    batch_im_a = np.array([im_a.transpose((2,0,1))])
    batch_im_b = np.array([im_b.transpose((2,0,1))])
    batch_label = np.array(int(label))

    batch_label = batch_label.reshape((-1, 1))
    batch_label = torch.from_numpy(batch_label).float()  # convert the numpy array into torch tensor
    batch_label = Variable(batch_label).cuda()           # create a torch variable and transfer it into GPU

    batch_im_a = torch.from_numpy(batch_im_a).float()     # convert the numpy array into torch tensor
    batch_im_a = Variable(batch_im_a).cuda()              # create a torch variable and transfer it into GPU

    batch_im_b = torch.from_numpy(batch_im_b).float()  # convert the numpy array into torch tensor
    batch_im_b = Variable(batch_im_b).cuda()           # create a torch variable and transfer it into GPU

    features_1 = model(batch_im_a)
    features_2 = model(batch_im_b)

    dist = F.pairwise_distance(features_1, features_2).cpu()
    dist = dist.data.numpy()[0]

    batch_a = Variable(batch_im_a).cuda()
    batch_b = Variable(batch_im_b).cuda()

    features_1 = model(batch_a)
    features_2 = model(batch_b)

    euclidean_distance = F.pairwise_distance(features_1, features_2)
    print euclidean_distance.data.cpu().numpy()[0]

    w = batch_label * torch.pow(euclidean_distance, 2)

    loss_contrastive = torch.mean(batch_label * torch.pow(euclidean_distance, 2) +
            (1 - batch_label) * torch.pow(torch.clamp(1.0 - euclidean_distance, min=0.0), 2))/2.0

    y = features_1.data.cpu().numpy()
    z = features_2.data.cpu().numpy()

        # print y

        # euclidean_distance = F.pairwise_distance(features_1, features_2)
        # print euclidean_distance.data.cpu().numpy()


    #######
    #caffe#
    #######


    def _get_transformer(shape, mean, lib='cf'):
        transformer = caffe.io.Transformer({'data':shape})
        transformer.set_transpose('data', (2, 0, 1))
        transformer.set_mean('data', mean)

        if lib == 'cf':
            transformer.set_raw_scale('data', 255)
            transformer.set_channel_swap('data', (2,1,0))

        return transformer

    pixel_means = np.array([[[104.0, 117.0, 123.0]]])
    net.blobs['data'].reshape(1, 3, 224, 224)
    transformer = _get_transformer(net.blobs['data'].data.shape, np.array([104, 117, 123]), 'cv')

    objectid_to_metadata = json.load(open(data_json, 'r'))
    pair_data = json.load(open(pair_json, 'r'))

    object_id1, object_id2, l = '239892', '239924', '90'

    object1 = objectid_to_metadata[object_id1]
    object2 = objectid_to_metadata[object_id2]

    print object1
    
    image1 = np.rot90(crop_image1(object1, int(l))).copy()
    image2 = np.rot90(crop_image1(object2, int(l))).copy()

    image1 -= pixel_means
    image2 -= pixel_means

    target_size = 224
    image1 = np.array([cv2.resize(image1, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)])

    image2 = np.array([cv2.resize(image2, (target_size, target_size),
                    interpolation=cv2.INTER_LINEAR)])
    channel_swap = (0, 3, 1, 2)
    image1 = image1.transpose(channel_swap)
    image2 = image2.transpose(channel_swap)

    # tmp = image1[0].transpose((1,2,0))
    # cv2.imwrite('a.jpg', tmp)

    # print x -image1

    # print image1.shape

    # image1, image2 = [transformer.preprocess('data', image1)], [transformer.preprocess('data', image2)]
    # tmp = image1[0].transpose((1,2,0))
    # cv2.imwrite('c.jpg', tmp)
    net.blobs['data'].data[...] = image1
    # print x - image1
    net.blobs['data_p'].data[...] = image2
    net.forward()
    feat = net.blobs['feat'].data.copy()
    feat_p = net.blobs['feat_p'].data.copy()
    dist = np.linalg.norm(feat - feat_p)
    print dist
    print feat - y
    print feat_p - z
    print np.sum(np.abs(feat - y))
    print np.sum(np.abs(feat_p - z))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Siamese")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--gpu_ids', type=int, default=0)
    parser.add_argument('--epoch_num', type=int, default=3)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)

    main(parser.parse_args())





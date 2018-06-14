import argparse
import os
import time
import json
import cv2
import sys

def cv2_loader(path):
    return cv2.imread(path)

def main(args):
    import caffe

    a = 'deploy.prototxt'
    b = '/core1/data/home/yanjia/models/caffe_siamese_model/resnet50_citybox_orig_v17_iter_118000.caffemodel'
    net = caffe.Net(a,b,caffe.TEST)

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
    import numpy as np

    torch.cuda.set_device(args.gpu_ids)
    normalize = cvtransforms.SubMean(mean=np.array([[[104.0, 117.0, 123.0]]]))

    # pair_json = '/data-4t/home/yanjia/siamese_shelf/length_data/json_data/multiprocess/pair_train_data_orig_mryx.json'
    pair_json = '../pretrain/pair_train.json'
    data_json = '/data-4t/home/yanjia/siamese_shelf/length_data/json_data/objectid_to_metadata_mryx_orig.json'
    image_json = '/home/zhuyanjia/siamese_zyj/data/image_data_mryx.json'

    train_dataset = ImageSiamese(
        pair_json,
        data_json,
        image_json,
        transforms.Compose([
            cvtransforms.Rot90(),
            cvtransforms.FixResize(224),
            normalize,
            cvtransforms.ToTensor(),
        ]), loader=cv2_loader)

    # print train_dataset.image_container['1222090']

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    model = res50_sia(pretrained=False,siamese=True)
    del model.base.fc
    model = nn.DataParallel(model,device_ids=[ args.gpu_ids ]).cuda()

    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    model.eval().cuda()


    for i, (batch_a, batch_b, batch_label) in enumerate(train_loader):
        print i
        x = batch_a.numpy()
        tmp = x[0].transpose((1,2,0))
        cv2.imwrite('b.jpg', tmp)
        print x.shape
        print batch_label[0].type()
        batch_label = Variable(batch_label.float()).cuda()
        print batch_label.type()
        batch_a = Variable(batch_a).cuda()
        batch_b = Variable(batch_b).cuda()

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

    object_id1, object_id2, l = pair_data.keys()[0].split(',')

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
    print image1[0].shape
    net.blobs['data'].data[...] = image1
    print net.blobs['data'].data.shape
    print x - image1
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





from __future__ import absolute_import

import sys
# sys.path.insert(0, '/core1/data/home/liuhuawei/github/pytorch_resource/inplace_abn')
# from get_net import get_net

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import time



__all__ = ['res50_sia']


class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, siamese=False, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, skip_connect=False,
                last_conv_stride=1, skip_classification=True):
        super(ResNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.skip_connect = skip_connect

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        if type(self.depth) is str and  's1' in self.depth:
            self.base = ResNet.__factory[depth](pretrained=pretrained, last_conv_stride=last_conv_stride, siamese=siamese)
        else:
            self.base = ResNet.__factory[depth](pretrained=pretrained, siamese=siamese)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            self.skip_classification = skip_classification
            out_planes = self.base.fc.in_features

            if self.skip_connect:
                out_planes = out_planes + \
                self.base.layer4[0].conv1.in_channels + \
                self.base.layer3[0].conv1.in_channels

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)


        # init.constant(self.base.conv1.bias, 0)
        # self.base.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        # init.xavier_normal(self.base.conv1.weight)
        # init.constant(self.base.conv1.bias, 0)        

        # del self.base.fc
        self.base.ip1 = nn.Linear(2048, 512)
        init.xavier_normal_(self.base.ip1.weight)
        init.constant_(self.base.ip1.bias, 0)
        self.base.ip2 = nn.Linear(512, 512)
        init.xavier_normal_(self.base.ip2.weight)
        init.constant_(self.base.ip2.bias, 0)
        self.base.feat = nn.Linear(512, 512)
        init.xavier_normal_(self.base.feat.weight)
        init.constant_(self.base.feat.bias, 0)


        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base._modules.items():
            if self.skip_connect and name=='layer3':
                pool_conv3_3 = x
                # print('shape of pool_conv3_3: ', pool_conv3_3.size())
            if self.skip_connect and name=='layer4':
                pool_conv4_3 = x    
                # print('shape of pool_conv4_3', pool_conv4_3.size())
            if name == 'avgpool':
                break
            x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.skip_classification:
            x = self.base.ip1(x)
            x = F.relu(x)
            x = self.base.ip2(x)
            x = self.base.feat(x)
            return x

        if self.skip_connect:
            pool_conv3_3 = F.avg_pool2d(pool_conv3_3, pool_conv3_3.size()[2:])
            pool_conv3_3 = pool_conv3_3.view(pool_conv3_3.size(0), -1)
            pool_conv4_3 = F.avg_pool2d(pool_conv4_3, pool_conv4_3.size()[2:])
            pool_conv4_3 = pool_conv4_3.view(pool_conv4_3.size(0), -1) 
            x = torch.cat((pool_conv4_3, pool_conv3_3, x), dim=1)
            # print('shape after concat: ', x.size())

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x = self.classifier(x)

        # raise
        return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

  

def res50_sia(**kwargs):
    return ResNet(50, **kwargs)








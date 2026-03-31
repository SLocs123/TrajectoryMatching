import torch
import torch.nn as nn
from .backbones.resnet_ibn_a import resnet50_ibn_a,resnet101_ibn_a,resnet152_ibn_a
from .backbones.resnext_ibn import resnext101_ibn_a
from .layers.pooling import GeM, GeneralizedMeanPooling,GeneralizedMeanPoolingP
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = model_name


        if model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)
            print('using resnet50_ibn_a as a backbone')
        elif model_name == 'resnet152':
            self.in_planes = 2048
            self.base = resnet152_ibn_a(last_stride=last_stride)
        elif model_name == 'resnet101_ibn_a':
            self.in_planes = 2048
            self.base = resnet101_ibn_a(last_stride, frozen_stages=cfg.MODEL.FROZEN)
            print('using resnet101_ibn_a as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if cfg.MODEL.POOLING_METHOD == 'gempoolP':
            print('using GeMP pooling')
            self.gap = GeneralizedMeanPoolingP()
        elif cfg.MODEL.POOLING_METHOD == 'gempool':
            print('using GeM pooling')
            self.gap = GeM(freeze_p=False)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes # what is this use for?
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE


        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None,cam_label=None):  # label is unused if self.cos_layer == 'no'
        if self.model_name =='efficientnet_b7':
            x = self.base.extract_features(x)
        else:
            x = self.base(x)
        global_feat = nn.functional.adaptive_avg_pool2d(x, (1, 1))

        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        # print("$$$$$$",feat)
        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path,map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i.replace('module.','')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_un_param(self, trained_path):
        param_dict = torch.load(trained_path,map_location='cpu')
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in self.state_dict():
            if 'classifier' in i or 'arcface' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))



def make_model(cfg, num_class, camera_num=0, view_num=0):
    print('===========ResNet===========')
    model = Backbone(num_class, cfg) # figure out what is num_class here
    return model
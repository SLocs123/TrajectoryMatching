import os
from ..reid_config import cfg
from .baseline.model import make_model


def build_reid_model(_mcmt_cfg):

    #  slices the string abs_file from the beginning up to (but not including) the last forward slash. 
    # the : refers to Start:End, so this means from start: end is the last slash position

    cfg.INPUT.SIZE_TEST = [384,384]
    cfg.MODEL.NAME = 'resnet101_ibn_a'
    model = make_model(cfg, num_class=100)
    model.load_param('feature_extractor/reid_model/resnet101_ibn_a_3.pth')

    return model,cfg

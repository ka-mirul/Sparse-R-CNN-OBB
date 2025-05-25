# Modified by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

# Original implementation by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.config import CfgNode as CN


def add_sparsercnn_obb_config(cfg):
    """
    Add config for SparseRCNN_OBB.
    """
    cfg.MODEL.RSparseRCNN = CN()
    cfg.MODEL.RSparseRCNN.NUM_CLASSES = 1
    cfg.MODEL.RSparseRCNN.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.RSparseRCNN.NHEADS = 8 #self_attn
    cfg.MODEL.RSparseRCNN.DROPOUT = 0.0
    cfg.MODEL.RSparseRCNN.DIM_FEEDFORWARD = 2048
    cfg.MODEL.RSparseRCNN.ACTIVATION = 'relu'
    cfg.MODEL.RSparseRCNN.HIDDEN_DIM = 256
    cfg.MODEL.RSparseRCNN.NUM_CLS = 1
    cfg.MODEL.RSparseRCNN.NUM_REG = 3
    cfg.MODEL.RSparseRCNN.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.RSparseRCNN.NUM_DYNAMIC = 2
    cfg.MODEL.RSparseRCNN.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.RSparseRCNN.CLASS_WEIGHT = 2.0
    cfg.MODEL.RSparseRCNN.IOU_WEIGHT = 2.0
    cfg.MODEL.RSparseRCNN.L1_WEIGHT = 5.0
    cfg.MODEL.RSparseRCNN.DEEP_SUPERVISION = True
    cfg.MODEL.RSparseRCNN.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.RSparseRCNN.USE_FOCAL = True
    cfg.MODEL.RSparseRCNN.ALPHA = 0.25
    cfg.MODEL.RSparseRCNN.GAMMA = 2.0
    cfg.MODEL.RSparseRCNN.PRIOR_PROB = 0.01

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.AMSGRAD = "True"

    cfg.DATASET_NAME = "RSDD" #RSDD or SSDD
    cfg.DATASET_DIR = "/home/mikicil/xo23898/SHIP_DETECTION/DATASET/RSDD-SAR"

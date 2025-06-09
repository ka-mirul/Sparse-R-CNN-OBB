# Modified by Kamirul Kamirul
# Contact: kamirul.apr@gmail.com

# Original implementation by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
import copy
from detectron2.layers import ShapeSpec
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, detector_postprocess
from detectron2.modeling.roi_heads import build_roi_heads

from detectron2.structures import Boxes, ImageList, Instances, RotatedBoxes
from detectron2.utils.logger import log_first_n
from fvcore.nn import giou_loss, smooth_l1_loss

from .loss import SetCriterion, HungarianMatcher
from .head import DynamicHead
from .util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
import cv2
import numpy as np
import time

__all__ = ["SparseRCNN_OBB"]


@META_ARCH_REGISTRY.register()
class SparseRCNN_OBB(nn.Module):
    """
    Implement SparseRCNN-OBB
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)

        self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.num_classes = cfg.MODEL.RSparseRCNN.NUM_CLASSES
        self.num_proposals = cfg.MODEL.RSparseRCNN.NUM_PROPOSALS
        self.hidden_dim = cfg.MODEL.RSparseRCNN.HIDDEN_DIM
        self.num_heads = cfg.MODEL.RSparseRCNN.NUM_HEADS

        # Build Backbone.
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        
        # Build Proposals.
        self.init_proposal_features = nn.Embedding(self.num_proposals, self.hidden_dim) # [100, 256]
        
        '''
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 4) #the shape would be : [100, 4]      
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2:], 1.0)
        '''

        #'''
        # OBB
        self.init_proposal_boxes = nn.Embedding(self.num_proposals, 5)
        nn.init.constant_(self.init_proposal_boxes.weight[:, :2], 0.50) # centre
        nn.init.constant_(self.init_proposal_boxes.weight[:, 2], 0.25) # wh
        nn.init.constant_(self.init_proposal_boxes.weight[:, 3], 0.50) # wh
        nn.init.constant_(self.init_proposal_boxes.weight[:, 4], -1.00) # 45/180 (-180 to 180)
        #'''
        
        
        # Build Dynamic Head.
        self.head = DynamicHead(cfg=cfg, roi_input_shape=self.backbone.output_shape())

        # Loss parameters:
        class_weight = cfg.MODEL.RSparseRCNN.CLASS_WEIGHT # 2.0 label
        iou_weight = cfg.MODEL.RSparseRCNN.IOU_WEIGHT #2.0 bbox
        l1_weight = cfg.MODEL.RSparseRCNN.L1_WEIGHT #5.0 bbox
        no_object_weight = cfg.MODEL.RSparseRCNN.NO_OBJECT_WEIGHT
        self.deep_supervision = cfg.MODEL.RSparseRCNN.DEEP_SUPERVISION
        self.use_focal = cfg.MODEL.RSparseRCNN.USE_FOCAL

        # Build Criterion.
        matcher = HungarianMatcher(cfg=cfg,
                                   cost_class=class_weight, 
                                   cost_bbox=l1_weight, 
                                   cost_iou=iou_weight,
                                   use_focal=self.use_focal)
        weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_iou": iou_weight}
        if self.deep_supervision: #True
            aux_weight_dict = {}
            for i in range(self.num_heads - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict) 

        losses = ["labels", "boxes"]

        self.criterion = SetCriterion(cfg=cfg,
                                      num_classes=self.num_classes,
                                      matcher=matcher,
                                      weight_dict=weight_dict,
                                      eos_coef=no_object_weight,
                                      losses=losses,
                                      use_focal=self.use_focal) #true

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)


    def forward(self, batched_inputs, do_postprocess=True):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances: Instances

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        """
        images, images_whwhwh = self.preprocess_image(batched_inputs) #include image normalization

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        # Feature Extraction.
        src = self.backbone(images.tensor)


        features = list()        
        for f in self.in_features:
            feature = src[f]
            features.append(feature)

        proposal_boxes = self.init_proposal_boxes.weight.clone() #[100,4]
        proposal_boxes = proposal_boxes[None] * images_whwhwh[:, None, :] #[2, 100,4] # 2 is batch zise, embeded from images_whwh

        # Prediction.
        outputs_class, outputs_coord = self.head(features, proposal_boxes, self.init_proposal_features.weight)
                
        output = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]} # get latest output
        

        if self.training:
            
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            targets = self.prepare_targets(gt_instances)
            if self.deep_supervision:
                output['aux_outputs'] = [{'pred_logits': a, 'pred_boxes': b}
                                         for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

            loss_dict = self.criterion(output, targets)


            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]


            return loss_dict
        

        else:

            box_cls = output["pred_logits"]
            box_pred = output["pred_boxes"]
            results = self.inference(box_cls, box_pred, images.image_sizes)
            
            if do_postprocess:
                processed_results = []
                for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                    height = input_per_image.get("height", image_size[0])
                    width = input_per_image.get("width", image_size[1])
                    r = detector_postprocess(results_per_image, height, width)
                    processed_results.append({"instances": r})
                return processed_results
            else:
                return results
        
            

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:


            target = {}
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h, 180], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            
            #gt_boxes = box_xyxy_to_cxcywh(gt_boxes)

            target["labels"] = gt_classes.to(self.device)
            target["boxes"] = gt_boxes.to(self.device) # normalized
            target["boxes_xyxy"] = targets_per_image.gt_boxes.tensor.to(self.device) #un-normalized
            target["image_size_xyxy"] = image_size_xyxy.to(self.device)
            image_size_xyxy_tgt = image_size_xyxy.unsqueeze(0).repeat(len(gt_boxes), 1)
            target["image_size_xyxy_tgt"] = image_size_xyxy_tgt.to(self.device)
            #target["area"] = targets_per_image.gt_boxes.area().to(self.device)
            new_targets.append(target)

        return new_targets

    def inference(self, box_cls, box_pred, image_sizes):
        """
        Arguments:
            box_cls (Tensor): tensor of shape (batch_size, num_proposals, K).
                The tensor predicts the classification probability for each proposal.
            box_pred (Tensor): tensors of shape (batch_size, num_proposals, 4).
                The tensor predicts 4-vector (x,y,w,h) box
                regression values for every proposal
            image_sizes (List[torch.Size]): the input image sizes

        Returns:
            results (List[Instances]): a list of #images elements.
        """
        assert len(box_cls) == len(image_sizes)
        results = []

        

        if self.use_focal:
            scores = torch.sigmoid(box_cls)
            labels = torch.arange(self.num_classes, device=self.device).\
                     unsqueeze(0).repeat(self.num_proposals, 1).flatten(0, 1)

            for i, (scores_per_image, box_pred_per_image, image_size) in enumerate(zip(
                    scores, box_pred, image_sizes
            )):
                result = Instances(image_size)
                scores_per_image, topk_indices = scores_per_image.flatten(0, 1).topk(self.num_proposals, sorted=False)
                labels_per_image = labels[topk_indices]
                box_pred_per_image = box_pred_per_image.view(-1, 1, 5).repeat(1, self.num_classes, 1).view(-1, 5)
                box_pred_per_image = box_pred_per_image[topk_indices]

                
                result.pred_boxes = RotatedBoxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        else:
            # For each box we assign the best class or the second best if the best on is `no_object`.
            scores, labels = F.softmax(box_cls, dim=-1)[:, :, :-1].max(-1)

            for i, (scores_per_image, labels_per_image, box_pred_per_image, image_size) in enumerate(zip(
                scores, labels, box_pred, image_sizes
            )):
                result = Instances(image_size)
                result.pred_boxes = Boxes(box_pred_per_image)
                result.scores = scores_per_image
                result.pred_classes = labels_per_image
                results.append(result)

        return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        #x['image'].shape = [3,640,640]
        images = [self.normalizer(x["image"].to(self.device)) for x in batched_inputs]
        images = ImageList.from_tensors(images, self.size_divisibility)

        images_whwhwh = list()
        for bi in batched_inputs:
            h, w = bi["image"].shape[-2:]
            images_whwhwh.append(torch.tensor([w, h, w, h, 180], dtype=torch.float32, device=self.device))
        images_whwhwh = torch.stack(images_whwhwh)

        return images, images_whwhwh

"""
Implements the Generalized R-CNN for PA2PAT
"""
from torch import nn

import ps2pat.operator_patch.run_operator_patch

from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.rpn.rpn import build_rpn

from .roi_heads import build_roi_heads
from .backbone.backbone_ext import build_backbone


class PS2PAT(nn.Module):
    """
    Main class for R-CNN. Currently supports boxes and tracks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and
             computes detections / tracks from it.
    """

    def __init__(self, cfg):
        super(PS2PAT, self).__init__()

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.track_memory = None

    def flush_memory(self, cache=None):
        self.track_memory = cache

    def reset_ps2pat_status(self):
        self.flush_memory()
        self.roi_heads.reset_roi_status()

    def forward(self, images, targets=None, given_detection=None):

        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)

        # proposal_losses: the losses for the model during training. 
        # During testing, it is an empty dict.
        proposals, proposal_losses = self.rpn(images, features, targets)

        if self.roi_heads:
            x, result, roi_losses = self.roi_heads(features,
                                                   proposals,
                                                   targets,
                                                   self.track_memory,
                                                   given_detection)
            if not self.training:
                self.flush_memory(cache=x)

        else:
            raise NotImplementedError

        if self.training:
            losses = {}
            losses.update(roi_losses)
            losses.update(proposal_losses)
            return result, losses

        return result


def build_ps2pat(cfg):
    pa2pat = PS2PAT(cfg)
    return pa2pat

"""
Various Network "heads" for classification and bounding box prediction.

The design is as follows:

... -> RoI ----\                               /-> box cls output -> cls loss
                -> RoIFeatureXform -> box head
... -> Feature /                               \-> box reg output -> reg loss
       Map
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.utils.blob as blob_utils

# Fast R-CNN outputs and losses

def add_fast_rcnn_outputs(model, blob_in, dim):

    # Box classification layer
    model.FC(
        blob_in,
        'cls_score',
        dim,
        model.num_classes,
        weight_init=gauss_fill(0.01),
        bias_init=const_fill(0.0)
    )
    if not model.train:
        # only add softmax when testing;during training the softmax is combined
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')

    # Box regression layer
    num_box_reg_classes = (2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes)
    model.FC(
        blob_in,
        'bbox_pred',
        dim,
        num_box_reg_classes * 4,
        weight_init=gauss_fill(0.001),
        bias_init=const_fill(0.0)
    )


def add_fast_rcnn_losses(model):
    """Add losses for ROI classification and bounding box regression."""
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=model.GetLossScale()
    )
    loss_bbox = model.net.SmoothL1Loss(
        [
            'bbox_pred', 'bbox_targets', 'bbox_inside_weights',
            'bbox_outside_weights'
        ],
        'loss_bbox',
        scale=model.GetLossScale()
    )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.AddLosses(['loss_cls', 'loss_bbox'])
    model.AddMetrics('accuracy_cls')
    return loss_gradients


# Box heads

def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    """Add a ReLU MLP with two hidden layers."""
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    model.FC(roi_feat, 'fc6', dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim

def add_roi_Xconv1fc_head(model, blob_in, dim_in, spatial_scale):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm."""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in,
        'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.Conv(
            current, 'head_conv'+ str(i+1), dim_in, hidden_dim, 3, stride=1, pad=1,
            weight_init=('MSRAFill', {}), bias_init=('ConstantFill', {'value':0.}),
            no_bias=0
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim
    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim


def add_roi_Xconv1fc_gn_head(model, blob_in, dim_in, spatial_scale):
    """with GroupNorm"""
    hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat', blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )
    current = roi_feat
    for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
        current = model.ConvGN(
            current, 'head_conv' + str(i+1), dim_in, hidden_dim, 3, group_gn=get_group_gn(hidden_dim),
            stride=1, pad=1, weight_init=('MSRAFill', {}), bias_init=('ConstantFill', {'value': 0.})
        )
        current = model.Relu(current, current)
        dim_in = hidden_dim
    fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    model.FC(current, 'fc6', dim_in * roi_size * roi_size, fc_dim)
    model.Relu('fc6', 'fc6')
    return 'fc6', fc_dim



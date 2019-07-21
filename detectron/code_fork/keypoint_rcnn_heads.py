from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils


# Keypoint R-CNN outputs and losses

def add_keypoint_outputs(model, blob_in, dim):
    """Add Mask R-CNN keypoint specific outputs: keypoint heatmaps."""
    # NxKxHxw
    upsampling_heatmap = (cfg.KRCNN.UP_SCALE > 1)

    if cfg.KRCNN.USE_DECONV:
        #
        blob_in = model.ConvTranspose(
            blob_in,
            'kps_deconv',
            dim,
            cfg.KRCNN.DECONV_DIM,
            kernel=cfg.KRCNN.DECONV_KERNEL,
            pad=int(cfg.KRCNN.DECONV_KERNEL / 2 -1),
            stride=2,
            weight_init=gauss_fill(0.01),
            bias_init=const_fill(0.0)
        )
        model.Relu('kps_deconv', 'kps_deconv')
        dim = cfg.KRCNN.DECONV_DIM

    if upsampling_heatmap:
        blob_name = 'kps_score_lowres'
    else:
        blob_name = 'kps_score'

    if cfg.KRCNN.USE_DECONV_OUTPUT:
        #  Use ConvTranspose to predict heatmaps; results in 2x upsampling
        blob_out = model.ConvTranspose(
            blob_in,
            blob_name,
            dim,
            cfg.KRCNN.NUM_KEYPOINTS,
            kernel=cfg.KRCNN.DECONV_KERNEL,
            pad=int(cfg.KRCNN.DECONV_KERNEL / 2 -1),
            stride=2,
            weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init=const_fill(0.0)
        )
    else:
        #
        blob_out = model.Conv(
            blob_in,
            blob_name,
            dim,
            cfg.KRCNN.NUM_KEYPOINTS,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
            bias_init = const_fill(0.0)
        )

    if upsampling_heatmap:
        # Increase heatmap output size via bilinear upsampling
        blob_out = model.BilinearInterpolation(
            blob_out, 'kps_score', cfg.KRCNN.NUM_KEYPOINTS,
            cfg.KRCNN.NUM_KEYPOINTS, cfg.KRCNN.UP_SCALE
        )
    return blob_out


def add_keypoint_losses(model):
    """Add Mask R-CNN keypoint specific losses."""
    # Reshape input from (N, K, H, W) to (NK, HW)
    model.net.Reshape(
        ['kps_score'], ['kps_score_reshaped', '_kps_score_old_shape'],
        shape=(-1, cfg.KRCNN.HEATMAP_SIZE * cfg.KRCNN.HEATMAP_SIZE)
    )

    kps_prob, loss_kps = model.net.SoftmaxWithLoss(
        ['kps_score_reshaped', 'keypint_locations_int32', 'keypoint_weights'],
        ['kps_prob', 'loss_kps'],
        scale=cfg.KRCNN.LOSS_WEIGHT / cfg.NUM_GPUS,
        spatial=0
    )
    if not cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
        model.StopGradient(
            'keypoint_loss_normalizer', 'keypoint_loss_normalizer'
        )
        loss_kps = model.net.Mul(
            ['loss_kps', 'keypoint_loss_normalizer'], 'loss_kps_normalized'
        )
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_kps])
    model.AddLosses(loss_kps)
    return loss_gradients


# Keypoint heads


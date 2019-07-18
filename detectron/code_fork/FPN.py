
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numpy as np

from detectron.core.config import cfg
from detectron.modeling.generate_anchors import generate_anchors
from detectron.utils.c2 import const_fill
from detectron.utils.c2 import gauss_fill
from detectron.utils.net import get_group_gn
import detectron.modeling.ResNet as ResNet
import detectron.utils.blob as blob_utils
import detectron.utils.boxes as box_utils

# Lowest and highest pyramid levels in the backbone network. For FPN, we assume
# that all networks have 5 spatial reductions, each by a factor of 2. Level 1
# would correspond to the input image, hence it does not make sense to use it.

LOWEST_BACKONE_LVL = 2  # E.g., "conv2"-like level
HIGHEST_BACKONE_LVL = 5  # E.g., "conv5"-like level

# FPN with ResNet

def add_fpn_ResNet50_conv5_body(model):
    return add_fpn_onto_conv_body(
        model, ResNet.add_ResNet50_conv5_body, fpn_level_info_ResNet50_conv5
    )

def add_fpn_ResNet50_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(

    )

def add_fpn_ResNet101_conv5_P2only_body(model):
    return add_fpn_onto_conv_body(

    )


# Functions for bolting FPN onto a backone architectures
def add_fpn_onto_conv_body(
        model, conv_body_func, fpn_level_info_func, P2only=False
):
    """Add the specified conv body to the model and then add FPN levels to it."""
    # Note: blobs_conv is in reversed order:[fpn5, fpn4, fpn3, fpn2]
    # similarly for dims_conv: [2048, 1024, 512, 256]
    # similarly fo spatial_scales_fpn: [1/32, 1/16, 1/8, 1/4]

    conv_body_func(model)
    blobs_fpn, dim_fpn, spatial_scales_fpn = add_fpn(
        model, fpn_level_info_func()
    )



def add_fpn(model, fpn_level_info):
    """Add FPN connections base on the model described in the FPN paper."""
    # FPN levels are built starting from the highest/coarest level of the
    # backbone (usually "conv5"). First we build down, recursively constructing
    # lower/finer resolution FPN levels. Then we build up, constructing levels
    # that are even higher/coarser than the starting level.
    fpn_dim = cfg.FPN.DIM
    min_level, max_level = get_min_max_levels()
    num_backbone_stages = (
        len(fpn_level_info.blobs) - (min_level - LOWEST_BACKONE_LVL)
    )
    lateral_input_blobs = fpn_level_info.blobs[:num_backbone_stages]
    output_blobs = [
        'fpn_inner_{}'.format(s)
        for s in fpn_level_info.blobs[:num_backbone_stages]
    ]
    fpn_dim_lateral = fpn_level_info.dims
    xavier_fill = ('XavierFill, {}')

    # For the coarsest bnackbone level:1x1 conv only seeds recursion
    if cfg.FPN.USE_GN:
        #
        c = model.model.ConGN(
            lateral_input_blobs[0],
            output_blobs[0], # note :this ia s prefix
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            group_gn=get_group_gn(fpn_dim),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
        output_blobs[0] = c #
    else:
        model.Conv(
            lateral_input_blobs[0],
            output_blobs[0],
            dim_in=fpn_dim_lateral[0],
            dim_out=fpn_dim,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=xavier_fill,
            bias_init=const_fill(0.0)
        )
    #
    # Step 1: recursively build down starting from the coarsest backbone level
    #

    # For other levels add top-down and lateral connections
    for i in range(num_backbone_stages - 1):
        add_topdown_lateral_module(
            model,
            output_blobs[i],
            lateral_input_blobs[i + 1],
            output_blobs[i+ 1],
            fpn_dim,
            fpn_dim_lateral[i + 1]
        )

    # Post-hoc scale-specific 3x3 convs
    blobs_fpn = []
    spatial_scales = []

def add_topdown_lateral_module(model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral):
    """Add a top-down lateral module."""
    # Lateral 1x1 conv
    if cfg.FPN.USE_GN:
        # use GroupNorm
        lat = model.ConvGN(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            group_gn=get_group_gn(dim_top),
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(const_fill(0.0) if cfg.FPN.ZERO_INIT_LATERAL
                         else ('XavierFill', {})),
            bias_init=const_fill(0.0)
        )
    else:
        lat = model.Conv(
            fpn_lateral,
            fpn_bottom + '_lateral',
            dim_in=dim_lateral,
            dim_out=dim_top,
            kernel=1,
            pad=0,
            stride=1,
            weight_init=(
                const_fill(0.0)
                if cfg.FPN.ZERO_INIT_LATERAL else ('XavierFill', {})
            ),
            bias_init=const_fill(0.0)
        )
    # Top-down 2x upsampling
    td = model.net.UpsampleNearest(fpn_top, fpn_bottom + '_topdown', scale=2)
    # Sum lateral and top-down
    model.net.Sum([lat, td], fpn_bottom)


def get_min_max_levels():
    """The min and max FPN levels required for supporting RPN and/or RoI
    transform operations on multiple FPN levels."""
    min_level = LOWEST_BACKONE_LVL
    max_level = HIGHEST_BACKONE_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    return min_level, max_level
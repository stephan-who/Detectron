from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.net import get_group_gn

# Bits for specific architectures

def add_ResNet50_conv4_body(model):
    return add_ResNet_convX_body(model, (3, 4, 6))


def add_ResNet50_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 4, 6, 3))

def add_ResNet101_conv4_body(model):
    return add_ResNet_convX_body(model, (3, 4, 24))

def add_ResNet101_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 4, 23, 3))


def add_ResNet152_conv5_body(model):
    return add_ResNet_convX_body(model, (3, 8, 36, 3))

# Generic ResNet components

def add_stage(
        model,
        prefix,
        blob_in,
        n,
        dim_in,
        dim_out,
        dim_inner,
        dilation,
        stride_init=2
):
    """
    Add a ResNet stage to the model by stacking n residual blocks.
    :param model:
    :param prefix:
    :param blob_in:
    :param n:
    :param dim_in:
    :param dim_out:
    :param dim_inner:
    :param dilation:
    :param stride_init:
    :return:
    """
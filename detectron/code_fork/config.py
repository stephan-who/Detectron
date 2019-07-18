"""Detectron config system.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from ast import literal_eval
from future.utils import iteritems
import copy, io,logging
import numpy as np
import os
import os.path as osp
import six

from detectron.utils.collections import AttrDict
from detectron.utils.io import cache_url

logger = logging.getLogger(__name__) # 日志模块

__C = AttrDict()

cfg = __C

# Training options
__C.TRAIN = AttrDict()

__C.TRAIN.WEIGHTS = ''

__C.TRAIN.DATASETS = ()

__C.TRAIN.SCALES = (600, )

__C.TRAIN.MAX_SIZE = 1000

__C.TRAIN.IMS_PER_BATCH = 2

# RoI minibatch size * per image * (number of regions of interest [ROIs]
# Total number of RoIs per training minibatch =
#    TRAIN.BATCH_SIZE_PER_IM * TRAIN.IMS_PER_BATCH * NUM_GPUS
__C.TRAIN.BATCH_SIZE_PER_IM = 64

# Target fraction of RoI minibatch that is labeled foreground
__C.TRAIN.FG_FRACTION = 0.25

__C.TRAIN.FG_THRESH = 0.5

# [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

__C.TRAIN.USE_FLIPPED = True

__C.TRAIN.BBOX_THRESH =0.5

# Divide by NUM_GPUS to determine actual period (e.g., 2000/8 => 2500 iters
# to allow for linear training schedule scaling
__C.TRAIN.SNAPSHOT_ITERS = 20000

# Proposal files must be in correspondence with the datasets listed in Train.DATASETS
__C.TRAIN.PROPOSAL_FILES = ()

# Make minibatches from images that have similar aspect ratios(i.e. both
# tall and thin or both short and wide)
# This feature is critical for saving memory ( and makes training slightly faster)
__C.TRAIN.ASPECT_GROUPING = True

# ---------------------------------------------------------------------------- #
# RPN training options
# ---------------------------------------------------------------------------- #

__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7

__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3

__C.TRAIN.RPN_FG_FRACTION = 0.5

# Total number of RPN examples per image
__C.TRAIN.RPN_BATCH_SIZE_PER_IM = 256

__C.TRAIN.RPN_NMS_THRESH = 0.7

# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used ,this is * per FPN level* (not total)
__C.TRAIN.RPN_PER_NMS_TOP_N = 12000


__C.TRAIN.RPN_POST_NMS_TOP_N = 2000

# Remove RPN anchors that go outside the image by RPN_STRADDLE_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
__C.TRAIN.RPN_STRADDLE_THRESH = 0

__C.TRAIN.PRN_MIN_SIZE = 0

__C.TRAIN.CROWD_FILTER_THRESH =0.7


# ------------------------------------------------------------------------------#
# FPN options
# ------------------------------------------------------------------------------#
__C.FPN = AttrDict()

__C.FPN.FPN_ON = False

__C.FPN.DIM = 256

__C.FPN.ZERO_INIT_LATERAL = False

# Stride of the coarest FPN level
__C.FPN.COARSEST_STRIDE = 32

# FPN mya be used for just RPN, just object detection, or both
__C.FPN.MULTILEVEL_ROIS = False
# Hyperparameters for the RoI-to-FPN level mapping heuristic
__C.FPN.ROI_CANONICAL_SCALE = 224
__C.FPN.ROI_CANONICAL_LEVEL = 4

__C.FPN.ROI_MAX_LEVEL = 5
__C.FPN.ROI_MIN_LEVEL = 2

__C.FPN.MULTILEVEL_RPN = False
__C.FPN.RPN_MAX_LEVEL = 6
__C.FPN.RPN_MIN_LEVEL = 2
__C.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)
# RPN anchors start at this case on RPN_MIN_LEVEL
# The anchor size doubled each level after that
# With a default of 32 and levels 2 to 6, we get anchor sizes of 32 to 512
__C.FPN.RPN_ANCHOR_START_SIZE = 32
__C.FPN.EXTRA_CONV_LEVELS = False

__C.FPN.USE_GN = False

# ------------------------------------------------------------------------------#
# Mask R-CNN options
# ------------------------------------------------------------------------------#


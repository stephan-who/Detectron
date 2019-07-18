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

logger = logging.getLogger(__name__)

__C = AttrDict()

cfg = __C

# Training options
__C.TRAIN = AttrDict()

__C.TRAIN.WEIGHTS = ''

__C.TRAIN.DATASETS = ()

__C.TRAIN.SCALES = (600, )

__C.TRAIN.MAX_SIZE = 1000

__C.TRAIN.IMS_PER_BATCH = 2
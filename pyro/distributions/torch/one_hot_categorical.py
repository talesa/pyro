from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.one_hot_categorical import OneHotCategorical as _OneHotCategorical
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import copy_docs_from


@copy_docs_from(_OneHotCategorical)
class OneHotCategorical(TorchDistribution):
    enumerable = True

    def __init__(self, ps=None, logits=None, *args, **kwargs):
        torch_dist = torch.distributions.OneHotCategorical(probs=ps, logits=logits)
        x_shape = logits.shape if ps is None else ps.shape
        event_dim = 1
        super(OneHotCategorical, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)

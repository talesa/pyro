from __future__ import absolute_import, division, print_function

import torch

from pyro.distributions.gamma import Gamma as _Gamma
from pyro.distributions.torch_wrapper import TorchDistribution
from pyro.distributions.util import broadcast_shape, copy_docs_from


@copy_docs_from(_Gamma)
class Gamma(TorchDistribution):
    reparameterized = True

    def __init__(self, alpha, beta, *args, **kwargs):
        torch_dist = torch.distributions.Gamma(alpha, beta)
        x_shape = torch.Size(broadcast_shape(alpha.size(), beta.size(), strict=True))
        event_dim = 1
        super(Gamma, self).__init__(torch_dist, x_shape, event_dim, *args, **kwargs)
        self.alpha = alpha
        self.beta = beta

    def sample(self, *args, **kwargs):
        return super(Gamma, self).sample(*args, **kwargs).clamp_(min=1.0e-35, max=1.0e30)

    def relative_entropy(self, target):
        term1 = (self.alpha - target.alpha) * torch.digamma(self.alpha)
        term2 = torch.lgamma(target.alpha) - torch.lgamma(self.alpha)
        term3 = target.alpha * (torch.log(self.beta) - torch.log(target.beta))
        term4 = (self.alpha / self.beta) * (target.beta - self.beta)
        return 1.0e-7 * (term1 + term2 + term3 + term4).sum()

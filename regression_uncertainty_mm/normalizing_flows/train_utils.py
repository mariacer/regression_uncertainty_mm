#!/usr/bin/env python3
# Copyright 2020 Rafael Daetwyler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :normalizing_flows/train_utils.py
# @author         :rd
# @contact        :rafael.daetwyler@uzh.ch
# @created        :11/24/2020
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for training scripts.
"""
import numpy as np
from scipy.stats import shapiro
import torch

from data.data_utils import _ONE_D_MISC as _MISC

from probabilistic import prob_utils as putils

def compute_nf_loss(flow, Y, logabsdet, reduction='mean'):    
    r"""Returns the loss of the normalizing flow.

    This function deals with a single model, and the sum for the MC estimate is
    therefore done outside this function.
    
    Overall, we want to minimize the expectation over our training samples of
    the KL divergence between our groundtruth :math:`p(y \mid x)` and our
    predictions :math:`q(y \mid x, w)`, so we want to minimize:

    .. math::

        \mathbb{E}_{p(x)}[D_{KL}(p(y|x) \parallel q(y|x,w))] &= \
            \mathbb{E}_{p(x)}\big[\mathbb{E}_{p(y|x)}[\log \
            \frac{p(y|x)}{q(y|x,w)}] \big] \\

    Dropping the terms that don't depent on the posterior we obtain that we
    want to minimize the negative cross-entropy 
    :math:`-\mathbb{E}_{p(x, y)}[\log q(y|x,w)] \big]`.

    In a normalizing flow, we have the following expression for
    :math:`q(y|x,w)`:

    .. math::
        
        q(y|x,w) = p_u(u) \mid \det \mathcal{J}_{T(x)}(u) \mid ^{-1}

    Furthermore we have that :math:`y = T_x(u)` so :math:`u = T_x^{-1}(y)`.
    Therefore :math:`p_u(u) = p_u(T_x^{-1}(y))`, and be obtain:

    .. math::

        - \mathbb{E}_{p(x, y)}[\log q(y|x,w)] \big] =
            - \mathbb{E}_{p(x, y)}[\log p_u(T_x^{-1}(y)) + \
            \log \mid \det \mathcal{J}_{T_x}(T_x^{-1}(y)) \mid ^{-1}]

    Noticing that
    :math:`\mid \det \mathcal{J}_{T_x}(T_x^{-1}(y)) \mid ^{-1} = \mid \det \
    \mathcal{J}_{T_x^{-1}}(y) \mid` we obtain the following MC estimate and
    loss function:

    .. math::

        L(y, x) = - \frac{1}{n}\sum_{i=1}^n [ \log p_u(T_{x_i}^{-1}(y_i)) + \
            \log \mid \det \mathcal{J}_{T_{x_i}^{-1}}(y_i) \mid ]

    Args:
        flow (SimpleFlow): The normalizing flow.
        Y: Output tensor consisting of means. Same shape as ``T``.
        logabsdet: The log of the absolute of the Jacobian.

    Returns:
        (float): The loss.
    """
    assert Y.shape == logabsdet.shape or (len(Y.shape) == 2 and Y.shape[1] == 2)
    if reduction == 'mean':
        return - torch.mean(flow.log_p0(Y) + logabsdet)
    else:
        return - torch.sum(flow.log_p0(Y) + logabsdet)

def normality_test(data, flow, hnet, config, device, x=None, sample_size=100,
                   return_samples=False):
    """Performs the Shapiro–Wilk test on the null hypothesis that the likelihood
    is normally distributed at a training point.

    See SciPy
    `_documentation <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html>`_.

    Args:
        data: Data handler.
        flow: Normalizing flow.
        hnet: Hypernetwork.
        device: CUDA device.
        config: The experiment config.
        x (optional): A point at which to test the normality of the likelihood.
        sample_size: Sample size to use for the normality test.
        return_samples: If True, the samples used to compute the p-value are
            also returned.

    Returns:
        (float): p-value of the test.
    """
    if x is None:
        x = data.input_to_torch_tensor(np.array([_MISC.x]), device)

    if config.mean_only:
        w_mean = hnet.weights
    else:
        w_mean, w_rho = hnet.extract_mean_and_rho(weights=None)

    if config.mean_only or sample_size == 1:
        sample_size = 100

    # Iterate over the number of models.
    if not config.mean_only:
        for j in range(sample_size):
            weights_j = hnet.forward(x, weights=None, mean_only=False,
                            extracted_mean=w_mean, extracted_rho=w_rho,
                            ret_format='flattened')
            if j == 0:
                weights = torch.zeros(sample_size,weights_j.shape[1]).to(device)
            weights[j, :] = weights_j
    else:
        weights = hnet.forward(x, ret_format='flattened')
        weights = weights.expand(sample_size, -1)

    samples_pu = flow.sample_p0(sample_size)
    samples_py, _ = flow.inverse(samples_pu, weights)

    samples_py_numpy = samples_py.detach().cpu().numpy()
    statistic, p_value = shapiro(samples_py_numpy)

    if return_samples:
        return p_value, samples_py_numpy

    return p_value


def max_deviation(data, flow, hnet, device, x=None):
    """Computes the maximum deviation between forward and inverse passes of a
    normalizing flow.

    Args:
        (....): See docstring of function :func:`normality_test`.

    Returns:
        (float): The maximum deviation.
    """
    x_temp = None
    if x is None:
        x_tmp = _MISC.x

    train_data = data.get_train_outputs()[:200]
    train_data = data.output_to_torch_tensor(train_data, device)

    x = data.input_to_torch_tensor(np.array([x_tmp]), device)
    weights = hnet.forward(x, ret_format='flattened')
    weights = weights.expand(len(train_data),-1)

    return flow.get_max_deviation(train_data, weights)


def compute_entropy_nf(mnet, hnet, config, x_torch, num_inputs,
                       num_samples=1000, num_models=100):
    """Compute the entropy of a posterior predictive distribution based on NFs.

    For details about the value being compute please refer to
    :func:`gaussian_likelihoods.train_utils.compute_entropy_gaussian`.
    
    Args:
        mnet: The main network.
        hnet: The hypernetwork.
        config: The configuration.
        x_torch (torch.Tensor): The inputs.
        num_inputs (int): Size of the input range.
        num_samples (int): The number of samples to use for each input.
        num_models (int): The number of models to sample for Bayesian networks.

    Returns:
        (np.array): The averaged entropy across weight samples for the entire
            input range.
    """
    # Extract Bayesian model parameters.
    if config.mean_only:
        w_mean = hnet.weights
        w_std = None
    else:
        w_mean, w_rho = hnet.extract_mean_and_rho(weights=None)

    ### First, sample y's from the posterior predictive distribution.
    weights = []
    for n in range(num_samples):
        if config.mean_only:
            weights.append(hnet.forward(x_torch, weights=w_mean,
                           ret_format='flattened'))
        else:
            weights.append(hnet.forward(x_torch, weights=None,
                           mean_only=False, extracted_mean=w_mean,
                           extracted_rho=w_rho, ret_format='flattened'))
    weights = torch.stack(weights)
    z_sample = mnet.sample_p0(num_samples * num_inputs).flatten()
    weights = weights.reshape(-1, weights.shape[-1])
    y_samples, _ = mnet.inverse(z_sample, weights)
    y_samples = y_samples.reshape(num_samples, num_inputs) # num_samples,x_range

    ### Second, compute the densities of the above samples.
    weights = []
    for k in range(num_models):
        if config.mean_only:
            weights.append(hnet.forward(x_torch, weights=w_mean,
                           ret_format='flattened'))
        else:
            weights.append(hnet.forward(x_torch, weights=None,
                                        mean_only=False,
                                        extracted_mean=w_mean,
                                        extracted_rho=w_rho,
                                        ret_format='flattened'))
    weights = torch.stack(weights)
    weights = weights.expand(num_samples, -1, -1, -1).permute(2, 0, 1, 3)
    weights = weights.reshape(-1, weights.shape[-1])
    # extend y_samples to match shape of weights
    y_samples = y_samples.expand(num_models, -1, -1).permute(2, 1, 0)
    y_samples = y_samples.reshape(-1)

    z_val, logabsdet = mnet.forward(y_samples, weights)
    py = torch.exp(mnet.log_p0(z_val) + logabsdet)
    py = py.reshape(num_inputs, num_samples, num_models)
    py = py.detach().cpu().numpy() # x_range, num_samples, num_models
    logmean = np.log(np.mean(py, axis=2)) # average across models
    entropy = - np.mean(logmean, axis=1) # average across samples

    return entropy

if __name__ == '__main__':
    pass


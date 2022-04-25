#!/usr/bin/env python3
# Copyright 2021 Maria Cervera
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
# @title          :gaussian_likelihoods/train_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :20/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for training
-----------------------------

A collection of helper functions for training to keep other scripts clean.
"""
import numpy as np
import os
import sys
import torch
from torch.distributions.normal import Normal
import torch.nn.functional as F

def compute_nll(T, Y, sigma=None, ndims=1):
    r"""Compute the negative log-likehood of a Gaussian.

    It computes the following quantity:

    .. math::

        - \log q(y | x, w) &=  - \log \mathcal{N} (\mu, \sigma^2) \\
            &= - \log \Big[ \frac{1}{\sqrt{2 \pi} \sigma} exp \big( - \
            \frac{(y - \mu)^2}{2 \sigma^2}\big) \Big] \\
            &= \frac{1}{2} \log (2 \pi) + \log \sigma + \
            \frac{(y - \mu)^2}{2 \sigma^2}

    Assuming i.i.d. data, :math:`q(y | x, w)` is a product of the probabilities
    of individual samples, which corresponds to a sum in log scale.

    Args:
        T: The target tensor. It has shape ``[batch_size, 1]``
            since the loss is comupted separately for each weight sample.
        Y: Output tensor consisting of means. Same shape as ``T`` for cases
            with homoscedastic variance, else it will have a second column with
            the variances.
        sigma (float, optional): The standard deviation of the predictive
            distribution. Only provided for Gaussian models with homoscedastic
            variance.
        ndims (int): Dimensionality of the output space.

    Returns:
        (float): The negative log-likelihood.
    """

    if ndims == 1:
        assert sigma is not None or Y.shape[1] == 2
        homoscedastic = Y.shape[1] == 1

        if len(T.shape) > 1:
            assert T.shape[1] == 1
            T = T.squeeze()

        # Extract mean and variance.
        mu = Y[:, 0]
        if len(mu.shape) > 1:
            assert mu.shape[1] == 1
            mu = mu.squeeze()
        if not homoscedastic:
            rho = Y[:, 1]  
            sigma = torch.exp(1/2 * rho)
        else:
            sigma = sigma * torch.ones_like(mu).to(Y.device)
        if len(sigma.shape) > 1:
            assert sigma.shape[1] == 1
            sigma = sigma.squeeze()

        assert mu.shape == T.shape == sigma.shape
        nll = 0.5 * np.log(2 * np.pi) + torch.log(sigma) + \
            0.5 * torch.square((mu - T)/sigma)

        return nll.sum()

    elif ndims == 2:
        assert sigma is not None or Y.shape[1] == 5
        homoscedastic = Y.shape[1] == 2
        mu = Y[:,:2]
        if not homoscedastic:
            var = torch.exp(Y[:, 2:4])
            rho = torch.tanh(Y[:, 4]) # correlation factor

        else:
            # We assume a circular gaussian 
            # of radius the config argument --pred_dist_std.
            var = (sigma**2) * torch.ones((mu.shape[0],2)).to(Y.device)
            rho = torch.zeros((mu.shape[0]))
            # rho = torch.ones((mu.shape[0])) * 0.13


        T_c = T - mu
        det = (1 - rho**2)*var[:,0]*var[:,1]
        nll = np.log(2*np.pi) + 0.5 * torch.log(det) + \
            0.5/ det * (var[:,1]*(T_c[:,0]**2) + var[:,0]*(T_c[:,1]**2) \
                - 2 * rho * torch.sqrt(var[:,0]*var[:,1]) * T_c[:,0] * T_c[:,1])
        return nll.sum()

def compute_mse(T, Y, ndims=1):
    r"""Returns the MSE between predictions and targets.

    This function deals with a single model, and the sum for the MC estimate is
    therefore done outside this function.

    This function should be used as loss function for models with homoscedastic
    Gaussian likelihood, as it does not allow to learn the variances.

    This function simply computes:

    .. math::

        \frac{1}{n}\sum_{i=1}^n (y_i-\mu(x_i))^2

    It can be derived from the loss expression found in :func:`compute_loss`.

    .. math::

        L(y, \mu(x), \sigma(x)) = \frac{1}{n}\sum_{i=1}^n\left(\log \
            (\sigma(x_i)) + \frac{1}{2}\left(\frac{y_i-\mu(x_i)}{\
            \sigma(x_i)}\right)^2 \right)

    Since the variances are not learned in this setting, 
    :math:`\sigma(x_i) = \sigma`, and the term :math:`\log (\sigma(x_i))` is
    constant w.r.t. the learnable parameters and can be dropped.
    We therefore obtain:

    .. math::

        L(y, \mu(x), \sigma(x)) = \frac{1}{n}\sum_{i=1}^n\left(\frac{1}{2} \
            \left(\frac{y_i-\mu(x_i)}{\sigma}\right)^2 \right)

    which is simply the MSE scaled by :math:`2 \sigma^2`, where :math:`\sigma`
    corresponds to our parameter ``config.pred_dist_std``.

    Note that it can also be derived from the KL between two gaussians:

    .. math::

        \mathbb{E}_{p(x)}[D_{KL}(p(y \mid x) \mid\mid q(y \mid x,w))] = \
            \mathbb{E}_{p(x)}[\log(\frac{\sigma}{\sigma_p})+ \
            \frac{\sigma_p^2 + (\mu_p(x) - \mu(x))^2}{2 \sigma^2} - \frac{1}{2}]

    where :math:`p(y \mid x)` is the groundtruth, :math:`q(y \mid x, w)`
    is the predictive distribution, and :math:`\mu_p(x)` and :math:`\sigma_p`
    are the input-dependent means and the fixed standard deviation of the
    groundtruth. Again, since only the means are learned we can drop constant
    terms and obtain the same expression as above.

    Args:
        T: The target tensor. It has shape ``[batch_size, 1]``
            since the loss is comupted separately for each weight sample.
        Y: Output tensor consisting of means. Same shape as ``T``.
        ndims (int): Dimensionality of the output space.

    Returns:
        (float): The MSE.
    """
    if ndims == 1:
        if len(Y.shape) > 1:
            assert Y.shape[1] == 1
            Y = Y.squeeze()
        if len(T.shape) > 1:
            assert T.shape[1] == 1
            T = T.squeeze()
        assert Y.shape == T.shape
        kl = torch.square(Y - T)

    elif ndims == 2:   
        assert Y.shape == T.shape     
        kl = torch.square(Y - T).squeeze()
        # Sum over vector units if multidimensional before reduce.
        kl = kl.sum(-1)

    return kl.mean()
    
def compute_cross_entropy(T, Y, sigma=None, reduction='mean',ndims = 1):
    r"""Returns the regression cross entropy loss.

    This function deals with a single model, and the sum for the MC estimate is
    therefore done outside this function.

    This function should be used as loss function for models with
    heteroscedastic Gaussian likelihood, as it allows to learn the variances.
    This function is derived by minimizing the KL divergence between ground
    truth and Gaussian model likelihood (for a derivation see dosctring of
    function :func:`compute_loss`). It computes the following loss:

    .. math::

        L(y, \mu(x), \sigma(x)) = \frac{1}{n}\sum_{i=1}^n\left(\log \
            (\sigma(x_i)) + \frac{1}{2}\left(\frac{y_i-\mu(x_i)}{\
            \sigma(x_i)}\right)^2 \right)

    Note that it can also be derived from the KL between two gaussians:

    .. math::

        \mathbb{E}_{p(x)}[D_{KL}(p(y|x) \parallel q(y|x,w))] &= \
            \mathbb{E}_{p(x)} \big[\log(\frac{\sigma(x)}{\sigma_p(x)}) + \
            \frac{\sigma_p(x)^2 + (\mu_p(x) - \mu(x))^2}{2 \sigma(x)^2} \
            - \frac{1}{2} \big] \\
            &= \mathbb{E}_{p(x)} \big[\log(\frac{\sigma(x)}{\sigma_p(x)}) + \
            \frac{\sigma_p(x)^2 + \mu_p(x)^2 - 2\mu_p(x)\mu(x)+ \mu(x)^2}{2 \
            \sigma(x)^2} - \frac{1}{2} \big]

    where :math:`p(y \mid x)` is the groundtruth, :math:`q(y \mid x, w)`
    is the predictive distribution, and :math:`\mu_p(x)` and :math:`\sigma_p(x)`
    are the input-dependent means and standard deviations of the groundtruth.
    Now for machine learning applications we assume we can't have
    access to the groundtruth, so we don't have access to :math:`\mu_p(x)`,
    instead we have access to observations 
    :math:`y(x) = \mu_p(x) + \epsilon \sigma_p(x)` where
    :math:`\epsilon \sim \mathcal{N}(0, 1)`.

    .. math::

        \mathbb{E}_{p(x)}[D_{KL}(p(y|x) \parallel q(y|x,w))] &= \
            \mathbb{E}_{p(x)} \big[\log(\frac{\sigma(x)}{\sigma_p(x)}) + \
            \frac{\mathbb{E}_{p(y|x)}[y^2] - 2\mathbb{E}_{p(y|x)}[y] \mu(x)+ \
            \mu(x)^2}{2 \sigma(x)^2} - \frac{1}{2} \big] \\
            &= \mathbb{E}_{p(x)} \big[\log(\frac{\sigma(x)}{\sigma_p(x)}) + \
            \frac{\mathbb{E}_{p(y|x)}[y^2 - 2 y \mu(x)+ \
            \mu(x)^2]}{2 \sigma(x)^2} - \frac{1}{2} \big] \\
            &= \mathbb{E}_{p(x, y)} \big[\log(\frac{\sigma(x)}{\sigma_p(x)}) + \
            \frac{(y - \mu(x))^2}{2 \sigma(x)^2} - \frac{1}{2} \big] \\

    where we have used that :math:`\mathbb{E}_{p(y|x)}[y] = \mu_p(x)` and
    :math:`\mathbb{E}_{p(y|x)}[y^2] = var(y) + \mathbb{E}_{p(y|x)}[y]^2` so
    :math:`\mathbb{E}_{p(y|x)}[y^2] = \sigma_p(x)^2 + \mu_p(x)^2`. Recall
    that :math:`\sigma_p(x)` is unknown, and drops from the optimization, so
    we recover the expression above when doing an MC estimate.

    In practice, since entorpy-related terms drop, for optimization we only care
    about the negative log-likelihood term up to some constants.

    Args:
        (....): See docstring of function :func:`compute_mse`.
        ndims: Whether regression output is 1D or 2D.
        sigma (float, optional): The standard deviation of the predictive
            distribution. Only provided for Gaussian models with homoscedastic
            variance.

    Returns:
        The cross-entropy.
    """
    ce = compute_nll(T, Y, sigma=sigma, ndims=ndims)

    if reduction == 'mean':
        ce /= T.shape[0]
    
    return ce

def compute_entropy_gaussian(mnet, config, x_torch, num_samples=1000,
                             num_models=100):
    r"""Compute the entropy of a Gaussian posterior predictive distribution.

    We consider a Bayesian setting with a posterior predictive distribution:

    .. math :: 

        p(y|D,x) &= \int_w p(y|w,x)p(w |D) dw \\
            &\approx \frac{1}{K}\sum_{k=1}^K p(y|w_k,x)

    The differential entropy of this posterior predictive is given by:

    .. math :: 

        h\big[p(y|D,x)\big] &= -\int_{\mathcal{Y}} p(y|D,x) \log \
            p(y|D,x) dy \\

    If we can draw samples from the posterior and both sample from and
    evaluate the likelihood :math:`p(y|w,x)`, we can estimate the differential
    entropy of the posterior predictive using a double Monte Carlo estimate as
    follows:

    .. math :: 

        h\big[p(y|D,x)\big] \approx - \frac{1}{N} \sum_{i=1}^N \log \
            \Big( \frac{1}{K}\sum_{k=1}^K p(y_i|w_k,x) \Big)

    where :math:`\{y_i\}_{i=1}^N` is a Monte Carlo sample from the posterior
    predictive (not from a specific model). A sample :math:`y` can be drawn from
    the posterior predictive as follows:

    .. math ::
        &\hat{w} \sim p(w|D)\text{, and then}\\
        &y \sim p(y|\hat{w},x)

    Args:
        mnet: The main network.
        config: The configuration.
        x_torch (torch.Tensor): The inputs.
        num_samples (int): The number of samples to use for each input.
        num_models (int): The number of models to sample for Bayesian networks.

    Returns:
        (np.array): The averaged entropy across weight samples for the entire
            input range.
    """
    # Extract Bayesian model parameters.
    if config.mean_only:
        w_mean = mnet.weights
        w_std = None
    else:
        w_mean, w_rho = mnet.extract_mean_and_rho(weights=None)

    ### First, sample y's from the posterior predictive distribution.
    predictions = []
    for n in range(num_samples):
        if config.mean_only:
            predictions.append(mnet.forward(x_torch, weights=w_mean))
        else:
            predictions.append(mnet.forward(x_torch, weights=None,
                           mean_only=False, extracted_mean=w_mean,
                           extracted_rho=w_rho))
    predictions = torch.stack(predictions)
    predictions_std = torch.ones_like(predictions[:, :, 0]) * \
        config.pred_dist_std
    if predictions.shape[2] == 2:
        rho = predictions[:, :, 1] 
        predictions_std = torch.exp(0.5 * rho) # std
    predictions = predictions[:, :, 0] # means
    y_samples = torch.normal(predictions, predictions_std)

    ### Second, compute the densities of the above samples.
    predictions = []
    for k in range(num_models):
        if config.mean_only:
            predictions.append(mnet.forward(x_torch, weights=w_mean))
        else:
            predictions.append(mnet.forward(x_torch, weights=None,
                           mean_only=False, extracted_mean=w_mean,
                           extracted_rho=w_rho))
    predictions = torch.stack(predictions)
    predictions_std = torch.ones_like(predictions[:, :, 0]) * \
        config.pred_dist_std
    if predictions.shape[2] == 2:
        rho = predictions[:, :, 1] # standard deviations
        predictions_std = torch.exp(0.5 * rho) # std
    predictions = predictions[:, :, 0] # means

    predictions = predictions.unsqueeze(2).repeat(1, 1, num_samples)
    predictions_std = predictions_std.unsqueeze(2).repeat(1, 1, num_samples)
    normal_models = Normal(predictions, predictions_std)
    prob = torch.exp(normal_models.log_prob(y_samples.T))
    # average across models
    logmean = np.log(np.mean(prob.detach().cpu().numpy(), axis=0))
    # average across samples
    entropy = - np.mean(logmean, axis=1)

    return entropy

if __name__ == '__main__':
    pass
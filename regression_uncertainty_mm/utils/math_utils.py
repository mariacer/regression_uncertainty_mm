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
# @title          :utils/math_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :20/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for math operations
------------------------------------

A collection of helper math functions.
"""
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline


def normal_dist(x, mu, sigma):
    """Compute probabilities according to a Gaussian distribution.

    Args:
        x (np.array): The range of values where to obtain the probabilities.
        mu: The mean.
        sigma: The variance.

    Return:
        (np.array): The probabilities.
    """
    if sigma == 0:
        return dirac_delta(x, mu)

    norm_dist = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * \
        np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    # Normalize so that it sums up to 1.
    return norm_dist/norm_dist.sum()


def dirac_delta(x, mu, eps=2):
    """Heuristic dirac delta function for zero standard deviation.

    Args:
        x (np.array): The range of values where to obtain the probabilities.
        mu: The mean.
        eps: The width of the dirac in number of array items.

    Return:
        (np.array): The probabilities.
    """
    #  for std=0
    ret = np.zeros_like(x)
    ret[np.abs(x - mu) < eps] = 1 / eps
    return ret


def normal_dist_2D(x, mu, cov):
    """Compute probabilities according to a Bivariate Gaussian distribution.

    Args:
        x (np.array): Array of positions where to obtain the probabilities. 
        First dimension corresponds to dimensions of the gaussian.
        mu: The mean.
        sigma: The variance.

    Return:
        (np.array): The probabilities.
    """
    cent_x = x-mu.reshape(-1, 1, 1)  # Better way of doing this?
    if cov.size == 3:
        # First two terms are variances
        # last is correlation factor
        cov = cov.squeeze()
        corr = cov[2]*np.sqrt(np.prod(cov[:2]))
        cov = np.array([[cov[0], corr], [corr, cov[1]]])
    return 1/(2*np.pi * np.sqrt(np.linalg.det(cov))) * np.exp(-0.5 *
                                                              np.einsum('i...,i...->...', cent_x,
                                                                        np.tensordot(np.linalg.inv(cov), cent_x, axes=(1, 0))))


def interpolate(xvals, vals, n=300):
    """
    Interpolates between the values of an input array with normalization.

    To be used over distribution functions.

    Args:
        xvals (np.array): Array of positions associated with vals.
        vals (np.array): Array of evaluations of the function at positions xvals.
        n (int): Number of uniformly placed evaluation positions in the 
            initial interval, where the function will be evaluated.

    Return:
        xvals_new (np.array): new sampling positions
        vals_smooth (np.array): interpolations at the positions xvals_new.
    """
    xvals_new = np.linspace(xvals.min(), xvals.max(), n)

    spl = make_interp_spline(xvals, vals, k=3)
    vals_smooth = spl(xvals_new)
    vals_smooth /= vals_smooth.sum()
    return xvals_new, vals_smooth

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
# @title          :oodregression/utils/network_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for building networks
--------------------------------------

A collection of helper functions for generating networks.
"""
from hypnettorch.hnets import HMLP
from hypnettorch.mnets.mlp import MLP
from hypnettorch.mnets import ResNet
import hypnettorch.utils.misc as misc
import hypnettorch.utils.sim_utils as sutils
from hypnettorch.utils import torch_ckpts as tckpt
import numpy as np
import os
from time import time
import torch

from data.regression2d_gaussian import BivariateToyRegression
from networks.gauss_mlp import GaussianMLP
from networks.gauss_mnet_interface import GaussianBNNWrapper
from networks.mnet_hnet_wrapper import MnetHnetWrapper
from normalizing_flows.simple_regression.simple_flow import SimpleFlow
from normalizing_flows.simple_regression.bivariate_flow import BivariateFlow
from normalizing_flows.simple_regression.diffeomorphisms.log_transform import \
    LogTransform
from normalizing_flows.simple_regression.diffeomorphisms.normalization import \
    Normalization

def generate_networks(config, logger, dhandler, device, mode, dataset):
    """Generate networks for the experiments.

    Args:
        config: Command-line arguments.
        logger: Console (and file) logger.
        data_handler: The datahandler. Needed to extract the number of
            inputs/outputs of the main network.
        device: Torch device.
        mode (str): The experiment mode.  
        dataset (str): The dataset bein used.  

    Returns:
        (tuple): Tuple containing:

        - **mnet**: The main network.
        - **hnet**: Optionally, the hypernetwork.
    """
    hnet = None
    if mode == 'gl' or mode == 'ghl':
        mnet = generate_mnet(config, logger, dhandler, device, mode, dataset)
    elif mode == 'nf':
        mnet, hnet = generate_nf_networks(config, logger, dhandler, device,
                                          dataset)

    return mnet, hnet

def generate_mnet(config, logger, data_handler, device, mode, dataset):
    r"""Create main network.

    The function will first create a normal MLP and then convert it into a
    network with Gaussian weight distribution by using the wrapper
    :class:`probabilistic.gauss_mnet_interface.GaussianBNNWrapper`.

    This function also takes care of weight initialization.
    ..note ::
        In the case of 2d output space and Gaussian Heteroscedastic Likelihood,
        The network outputs a 2D mean and 3D for the covariance matrix: 
        :math:`\sigma_1^2,\sigma_2^2` and :math:`\rho`.
        The covariance matrix is given by variance terms in diagonal and 
        extradiagonal terms are both :math:`\rho \sigma_1 \sigma_2`.

    Args:
        (....): See docstring of function :func:`generate_networks`.

    Returns:
        (mnet): Main network instance.
    """
    # Local reparametrization trick only available for Gaussian networks.
    assert not config.mean_only or not config.local_reparam_trick

    ### Determine the dimensions of the network.
    # Input.
    if len(data_handler.in_shape) > 2:
        # Steering angle dataset.
        in_shape = data_handler.in_shape
    else:
        # Toy Datasets.
        n_x = data_handler.in_shape[0]
        n_y = data_handler.out_shape[0]
        in_shape = [n_x]
    # Output.
    n_y = data_handler.out_shape[0]
    if mode == 'ghl':
        # With heteroscedastic noise, the network outputs mean and variance.
        if isinstance(data_handler,BivariateToyRegression):
            # Output space is 2D -> 2D mean and 3D cov params
            n_y = 5
        else:
            n_y *= 2
    out_shape = [n_y]

    ### Main network.
    logger.info('Creating main network ...')
    if config.local_reparam_trick:
        mlp_arch = misc.str_to_ints(config.mlp_arch)
        net_act = misc.str_to_act(config.net_act)
        mnet = GaussianMLP(n_in=in_shape[0], n_out=out_shape[0],
            hidden_layers=mlp_arch, activation_fn=net_act,
            use_bias=not config.no_bias, no_weights=False).to(device)
    else:
        mnet_kwargs = {}
        if dataset == 'steering_angle':
            mnet_kwargs = {'chw_input_format' : True}
        mnet =  sutils.get_mnet_model(config, config.net_type, in_shape,
                                      out_shape, device, no_weights=False,
                                      **mnet_kwargs)

    # Initiaize main net weights.
    mnet.custom_init(normal_init=config.normal_init,
                     normal_std=config.std_normal_init, zero_bias=True)

    # Convert main net into Gaussian BNN.
    orig_mnet = mnet
    if not config.mean_only:
        logger.debug('Created main network will be converted into a ' +
                     'Gaussian main network.')
        mnet = GaussianBNNWrapper(mnet, no_mean_reinit=config.keep_orig_init,
            logvar_encoding=config.use_logvar_enc, apply_rho_offset=True
            ).to(device)

    return mnet


def generate_nf_networks(config, logger, data, device, dataset):
    """Create main network (flow) and hypernetwork.

    Args:
        (....): See docstring of function :func:`generate_networks`.

    Returns:
        (tuple): Tuple containing:

        - **mnet**: Main network instance.
        - **hnet** (optional): Hypernetwork instance.
    """

    ### Main network.
    logger.info('Creating flow ...')
    is_bivariate = isinstance(data,BivariateToyRegression)
    if is_bivariate:
        conditioner_arch = misc.str_to_ints(config.conditioner_arch)
        permute = config.permute_flow_layers
        if isinstance(permute,str):
            if permute == 'None':
                permute = None
            else:
                permute = misc.str_to_ints(config.permute_flow_layers)

    if config.flow_layer_type == 'splines':
        # Dirty hack to make data fit to spline input domain
        y_processed_torch = data.output_to_torch_tensor(data._data['out_data'],
                                                  device)
        y_processed = y_processed_torch.detach().cpu().numpy()
        spline_bound = max(np.max(np.abs(y_processed)) + 0.1, 10)
        if 'get_output_scale_translation' in dir(data):
            in_fn = Normalization
            scale, translation = data.get_output_scale_translation()
            kwargs = {'scale': scale, 'translation': translation}
            normalized_data = in_fn.forward(y_processed_torch, \
                **kwargs).detach().cpu().numpy()
            spline_bound = max(np.max(np.abs(normalized_data)) + 0.1, 10)
        else:
            in_fn = None
            kwargs = {}
        if not is_bivariate:
            mnet = SimpleFlow(depth=config.flow_depth,
                            dimensionality=data.out_shape[0], layers='splines',
                            spline_bound=spline_bound, in_fn=in_fn,
                            out_fn=LogTransform, **kwargs).to(device)
        else:
            mnet = BivariateFlow(depth=config.flow_depth,
                            permute=permute, layers='splines',
                            conditioner_arch=conditioner_arch,
                            spline_bound=spline_bound, in_fn=in_fn,
                            out_fn=LogTransform, **kwargs).to(device)
    elif config.flow_layer_type == 'perceptron':
        if not is_bivariate:        
            mnet = SimpleFlow(depth=config.flow_depth,
                        dimensionality=data.out_shape[0], negative_slope=0.1).\
                    to(device)
        else:
            mnet = BivariateFlow(depth=config.flow_depth, permute=permute,
                        negative_slope=0.1).\
                    to(device)
    else:
        raise NotImplementedError(\
                'Flow layer type %s not implemented.' % config.flow_layer_type)

    ### Hypernet.
    logger.info('Creating hypernetwork ...')
    if dataset == 'steering_angle':
        # For steering angle prediction data, we need a hypernetwork that can
        # process images as inputs, and therefore needs to be convolutional.
        # For this we use a class that wraps together a Resnet that processes
        # the data and that has as downstream layer an HMLP.
        mnet_kwargs = {'chw_input_format' : True}
        H = config.hmlp_uncond_in_size
        mnet_aux =  sutils.get_mnet_model(config, config.net_type, data.in_shape,
                                          [H], device, no_weights=False,
                                          **mnet_kwargs)
        hnet_aux = sutils.get_hypernet(config, device, 'hmlp',
                                       mnet.param_shapes, 0,
                                       no_cond_weights=True, uncond_in_size=H)
        
        ### Initialize networks.
        if config.hyper_fan_init:
            # The input to the hypernetwork has zero mean (which is required by
            # the hyperfan init) if the interval of the training data is
            # symmetric around zero.
            assert data._train_inter[0] == -data._train_inter[1]
            logger.info('Applying hyperfan initialization ...')
            train_var = np.var(data._data['in_data'][data._data['train_inds']])
            hnet_aux.apply_hyperfan_init(method='in', use_xavier=False,
                                     uncond_var=train_var, mnet=mnet)
        else:
            apply_custom_hnet_init(config, logger, hnet_aux)
        hnet = MnetHnetWrapper(mnet_aux, hnet_aux).to(device)
    else:
        hnet = sutils.get_hypernet(config, device, 'hmlp',
                                   mnet.param_shapes, 0, no_cond_weights=True,
                                   uncond_in_size=data.in_shape[0])

        ### Initialize networks.
        if config.hyper_fan_init:
            # The input to the hypernetwork has zero mean (which is required by
            # the hyperfan init) if the interval of the training data is
            # symmetric around zero.
            assert data._train_inter[0] == -data._train_inter[1]
            logger.info('Applying hyperfan initialization ...')
            train_var = np.var(data._data['in_data'][data._data['train_inds']])
            hnet.apply_hyperfan_init(method='in', use_xavier=False,
                                     uncond_var=train_var, mnet=mnet)
        else:
            apply_custom_hnet_init(config, logger, hnet)

    # If required make the hypernetwork Bayesian without reinitializing weights.
    if not config.mean_only:
        logger.info('Converting hypernetwork to BNN ...')
        hnet = GaussianBNNWrapper(hnet, no_mean_reinit=True).to(device)

    return mnet, hnet

def apply_custom_hnet_init(config, logger, hnet):
    """Applying a custom network init to the hnet (based on user configs).

    Note, this method might not be safe to use, as it is not customized to
    network internals.

    Args:
        config: Command-line arguments.
        logger: Console (and file) logger.
        hnet: Hypernetwork object.
    """
    assert (not hasattr(config, 'custom_network_init'))

    # FIXME temporary solution.
    logger.warning('Applying unsafe custom network initialization. This init ' +
                   'does not take the class internal structure into account ' +
                   'and fails, for instance, when using batch or spectral ' +
                   'norm.')

    init_params = list(hnet.parameters())

    for W in init_params:
        # FIXME not all 1D vectors are bias vectors (e.g., batch norm weights).
        if W.ndimension() == 1:  # Bias vector.
            torch.nn.init.constant_(W, 0)
        elif config.normal_init:
            torch.nn.init.normal_(W, mean=0, std=config.std_normal_init)
        else:
            torch.nn.init.xavier_uniform_(W)


def save_ckpts(config, score, mnet=None, hnet=None, iter=None, infos=None,
               mnet_ckpt_name='main_ckpt', hnet_ckpt_name='hypernet_ckpt'):
    """Wrapper to save checkpoints of networks.
    
    Args:
        config: Command-line arguments.
        score: See argument 'performance_score' of method
            'save_checkpoint'.
        mnet (optional): If given, the main network to checkpoint.
        hnet (optional): If given, the hypernetwork to checkpoint.
        iter (optional): A number that determines the checkpoint filename. See
            argument 'train_iter' of method 'save_checkpoint'.
        infos (optional): A dictionary, that should be saved in the checkpoints.
        mnet_ckpt_name (optional): Filename of main network checkpoint.
        hnet_ckpt_name (optional): Filename of hypernetwork checkpoint.
    """
    assert(hnet is not None or mnet is not None)
    assert(infos is None or 'state_dict' not in infos.keys())

    ts = time()

    ckpt_dir = os.path.join(config.out_dir, 'checkpoints')

    if hnet is not None:
        ckpt_dict = dict()
        if infos is not None:
            ckpt_dict = dict(infos)
        ckpt_dict['state_dict'] = hnet.state_dict(),
        tckpt.save_checkpoint(ckpt_dict,
                               os.path.join(ckpt_dir, hnet_ckpt_name),
                               score, train_iter=iter, timestamp=ts)

    if mnet is not None:
        ckpt_dict = dict()
        if infos is not None:
            ckpt_dict = dict(infos)
        ckpt_dict['state_dict'] = mnet.state_dict(),
        tckpt.save_checkpoint(ckpt_dict,
                               os.path.join(ckpt_dir, mnet_ckpt_name),
                               score, train_iter=iter, timestamp=ts)
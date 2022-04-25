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
# @title          :utils/train_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :25/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Utilities for running and training experiments
----------------------------------------------

Main training and testing functions for all experiments.
"""
from argparse import Namespace
import hypnettorch.utils.sim_utils as sutils
import numpy as np
import os
import pickle
from time import time
import torch

from data import data_utils
from data import plotting_utils as plt_utils
from data.regression2d_gaussian import BivariateToyRegression
from gaussian_likelihoods import train_utils
from gaussian_likelihoods.train_utils import compute_mse, compute_nll
from networks.gauss_mnet_interface import GaussianBNNWrapper
import networks.network_utils as net_utils
from normalizing_flows.train_utils import compute_nf_loss
import probabilistic.prob_utils as putils
from utils import sim_utils

def run(mode='gl', dataset='toy_1D'):
    """Run the script.

    Args:
        mode (str, optional): The mode of the experiment.
        dataset (str, optional): The dataset to be used.

    Returns:
        (float): The final loss.
    """
    assert mode in ['gl', 'ghl', 'nf']
    assert dataset in ['toy_1D', 'toy_2D', 'steering_angle']

    if mode in ['gl', 'ghl']:
        from gaussian_likelihoods import train_args
        from gaussian_likelihoods.train import train
    elif mode == 'nf':
        from normalizing_flows import train_args
        from normalizing_flows.train import train
    else:
        raise ValueError('Experiment mode %s unknown.' % mode)

    ### Start simulation.
    script_start = time()
    config = train_args.parse_cmd_arguments(mode=mode, dataset=dataset)
    device, writer, logger = sutils.setup_environment(config, logger_name=mode)
    sim_utils.backup_cli_command(config)
    if not config.no_plots:
        os.makedirs(os.path.join(config.out_dir,'figures'))

    ### Create the task.
    dhandler, dloaders = data_utils.generate_task(config, writer, dataset)
    assert not (dloaders is None and dataset == 'steering_angle')

    ### Create the networks.
    mnet, hnet = net_utils.generate_networks(config, logger, dhandler, device,
                                             mode, dataset)

    ### Simple struct, that is used to share data among functions.
    shared = sim_utils.setup_shared(config, device, mnet, hnet=hnet)

    ### Initialize the performance measures that are tracked during training.
    sim_utils.setup_summary_dict(config, shared, mnet, hnet=hnet)

    ### Train the network.
    logger.info('### Training ###')
    finished_training = train(dhandler, mnet, device, config, shared, logger,
          writer, mode, dataset, hnet, dloaders=dloaders)

    ### Finish the simulation.
    shared.summary['finished'] = 1 if finished_training else 0
    sim_utils.save_summary_dict(config, shared)
    writer.close()
    logger.info('Program finished successfully in %f sec.'
                % (time()-script_start))

    print('\nExplore tensorboard plots: ')
    print('tensorboard --logdir=%s'%config.out_dir)
    return shared.current_loss


def test(data, mnet, device, config, shared, logger, writer, mode, dataset,
         hnet=None, save_fig=True, plot=True, dloaders=None, epoch=None):
    """Test the performance.

    Args:
        (....): See docstring of method :func:`train`.
        save_fig: Whether the figures should be saved in the output folder.
        plot (bool): Whether to make plots.
        epoch (int): The number of epochs trained.

    Returns:
        (float): The current test loss value.
    """
    logger.info('### Testing ... ###')

    mnet.eval()

    disable_lrt_test = config.disable_lrt_test if \
        hasattr(config, 'disable_lrt_test') else None

    with torch.no_grad():

        # We want to measure loss values within the training range only!
        split_type = 'val'
        if data.num_val_samples == 0:
            split_type = 'train'
            logger.debug('Test - Using training set as no validation set is ' +
                         'available.')
        loss_val, val_struct = compute_loss(mode, data, mnet,
        								  device, config, shared, hnet=hnet,
                                          split_type=split_type,
                                          return_dataset=True,
                                          return_predictions=True,
                                          disable_lrt=config.disable_lrt_test,
                                          dloaders=dloaders,
                                          reduction='mean')
        loss = loss_val
        if mode == 'gl':
            mse = val_struct.mse_vals.mean()
        ident = 'validation'

        # Write summary metrics.
        writer.add_scalar('test/val_loss', loss, epoch)
        if mode == 'gl':
            writer.add_scalar('test/val_mse', mse, epoch)

        lowest_val = False
        shared.summary['aa_val_loss_final'] = loss_val.item()
        if shared.summary['aa_val_loss_lowest'] == -1 or \
                loss_val < shared.summary['aa_val_loss_lowest']:
            lowest_val = True
            shared.summary['aa_val_loss_lowest'] = loss_val.item()
            if not epoch is None:
                shared.summary['aa_val_loss_lowest_epoch'] = epoch

        # Record final mse, lowest mse and mse at lowest val.
        if mode == 'gl':
            shared.summary['aab_val_mse_final'] = mse.item()
            if shared.summary['aab_val_mse_lowest'] == -1:
                shared.summary['aab_val_mse_lowest'] = mse.item()
            elif mse < shared.summary['aab_val_mse_lowest']:
                shared.summary['aab_val_mse_lowest'] = mse.item()

        # For GHL, store validation standard deviations.
        if mode == 'ghl':
            with open(os.path.join(config.out_dir, 'val_std.pickle'), \
                                                            'wb') as handle:
                pickle.dump(val_struct.predictions_std, handle,
                            protocol=pickle.HIGHEST_PROTOCOL)

        # We want the predictive uncertainty across the whole test range!
        # We only do this if a test set is available.
        use_test_set = True
        if hasattr(data, 'test_angles_available') and not \
                data.test_angles_available:
            if not config.use_empty_test_set:
                use_test_set = False

        if use_test_set:
            loss_test, test_struct = compute_loss(mode, data, mnet,
                                            device, config, shared,
                                            split_type='test', hnet=hnet,
                                            return_dataset=True,
                                            return_predictions=True,
                                            disable_lrt=config.disable_lrt_test,
                                            dloaders=dloaders,
                                            reduction='mean')

            if dataset == 'toy_1D' and config.store_uncertainty_stats:
                 data_utils.compute_1D_uncertainty_stats(config, writer, data,
                                                         mnet, device,
                                                         hnet=hnet)

            # For GHL, store test standard deviations.
            if mode == 'ghl':
                with open(os.path.join(config.out_dir, 'test_std.pickle'), \
                                                                'wb') as handle:
                    pickle.dump(test_struct.predictions_std, handle,
                                protocol=pickle.HIGHEST_PROTOCOL)

            loss = loss_test
            if mode == 'gl':
                mse = test_struct.mse_vals.mean()
            ident = 'test'

            # Write summary metrics.
            writer.add_scalar('test/test_loss', loss, epoch)
            if mode == 'gl':
                writer.add_scalar('test/test_mse', mse, epoch)

            # Record final test loss and test loss at lowest val.
            shared.summary['aa_test_loss_final'] = loss_test.item()
            if lowest_val:
                shared.summary['aa_test_loss_at_lowest_val'] = loss_test.item()

            # Record lowest test loss.
            if shared.summary['aa_test_loss_lowest'] == -1:
                shared.summary['aa_test_loss_lowest'] = loss_test.item()
            elif loss_test < shared.summary['aa_test_loss_lowest']:
                shared.summary['aa_test_loss_lowest'] = loss_test.item()

            # Record final mse, lowest mse and mse at lowest val.
            if mode == 'gl':
                shared.summary['aab_test_mse_final'] = mse.item()
                if lowest_val:
                    shared.summary['aab_test_mse_at_lowest_val'] = mse.item()

                if shared.summary['aab_test_mse_lowest'] == -1:
                    shared.summary['aab_test_mse_lowest'] = mse.item()
                elif mse < shared.summary['aab_test_mse_lowest']:
                    shared.summary['aab_test_mse_lowest'] = mse.item()

    if plot and use_test_set:
        ### Plot predictive distributions over test range.
        fig_dir = os.path.join(config.out_dir,'figures')
        figname = None
        if save_fig:
            figname = os.path.join(fig_dir, 'test_predictions_epoch%i' % epoch)
        dloader = dloaders[2] if dloaders is not None else None
        plt_utils.plot_predictive_distribution(config, writer, data,
                                    test_struct, mnet, device, dataset,
                                    dloader=dloader, hnet=hnet, figname=figname,
                                    publication_style=config.publication_style,
                                    testing_mode='test')

    sim_utils.save_summary_dict(config, shared)

    logger.info('Test - Mean loss on %s set: %f' % (ident, loss))
    if mode == 'gl':
        logger.info('Test - MSE on %s set: %f' % (ident, mse))
    logger.info('### Testing ... Done ###')

    return loss

def evaluate(data, mnet, device, config, shared, logger, writer, mode, dataset,
             hnet=None, train_iter=None,  save_fig=False, plot=True,
             dloaders=None, epoch=None):
    """Evaluate the training progress.

    Evaluate the performance of the network on the validation set.
    Note, if no validation set is available, the test set will be used instead.

    Args:
        (....): See docstring of method :func:`train`.
        train_iter: The current training iteration. If not given, the `writer`
            will not be used.
    """
    if train_iter is None:
        logger.info('# Evaluating training ...')
    else:
        logger.info('# Evaluating network before running training step ' +
                    '%d ...' % (train_iter))

    if hnet is None:
        mnet.eval()
    else:
        hnet.eval()

    with torch.no_grad():
        # Note, if no validation set exists, we use the training data to compute
        # the loss (note, test data may contain out-of-distribution data in our
        # setup).
        split_type = 'train' if data.num_val_samples == 0 else 'val'
        if split_type == 'train':
            logger.debug('Eval - Using training set as no validation set is ' +
                         'available.')
        loss_val, val_struct = compute_loss(mode, data, mnet,
            device, config, shared, hnet=hnet, return_dataset=True,
            return_predictions=True, disable_lrt=config.disable_lrt_test,
            split_type=split_type, dloaders=dloaders, reduction='mean')
        ident = 'training' if split_type == 'train' else 'validation'
        logger.info('Eval - Mean loss on %s set: %f (std: %g).'
                    % (ident, loss_val, val_struct.loss_vals.std()))
        if mode == 'gl':
            logger.info('Eval - MSE on %s set: %f (std: %g).'
                        % (ident, val_struct.mse_vals.mean(), \
                           val_struct.mse_vals.std()))

        # In contrast, we visualize uncertainty using the test set.
        # We only do this if a test set is available.
        use_test_set = True
        if hasattr(data, 'test_angles_available') and not \
                data.test_angles_available:
            if not config.use_empty_test_set:
                use_test_set = False
        if use_test_set:
            loss_test, test_struct = compute_loss(mode, data, mnet,
                device, config, shared, hnet=hnet, return_dataset=True,
                return_predictions=True, disable_lrt=config.disable_lrt_test,
                split_type='test', dloaders=dloaders, reduction='mean')
            logger.info('Eval - Mean loss on test set: %f (std: %g).'
                         % (loss_test, test_struct.loss_vals.std()))
            if mode == 'gl':
                logger.info('Eval - MSE on test set: %f (std: %g).'
                            % (test_struct.mse_vals.mean(), \
                               test_struct.mse_vals.std()))

        if train_iter is not None:
            writer.add_scalar('eval/val_loss', loss_val, train_iter)
            if use_test_set:
                writer.add_scalar('eval/test_loss', loss_test, train_iter)

        if plot and (config.show_plots or train_iter != None):
            fig_dir = os.path.join(config.out_dir,'figures')

            # Plot val predictions.
            figname = None
            if save_fig:
                figname = os.path.join(fig_dir, 'val_predictions_epoch%i' \
                                       % epoch)
            dloader = dloaders[1] if dloaders is not None else None
            plt_utils.plot_predictive_distribution(config, writer, data,
                                    val_struct, mnet, device, dataset,
                                    dloader=dloader, hnet=hnet, figname=figname,
                                    train_iter=train_iter,
                                    publication_style=config.publication_style,
                                    testing_mode='eval')

            # Plot entropy.
            if dataset == 'toy_1D':
                figname = None
                if save_fig:
                    figname = os.path.join(fig_dir, 'val_entropy_epoch%i' \
                                           % epoch)
                plt_utils.plot_entropy(config, writer, data, val_struct, mnet,
                                     device, dataset, hnet=hnet,
                                     figname=figname, train_iter=train_iter,
                                     testing_mode='eval',
                                     publication_style=config.publication_style,
                                     num_models=config.val_sample_size)

            # Plot test predictions.
            if use_test_set:
                figname = None
                if save_fig:
                    figname = os.path.join(fig_dir, 'test_predictions_epoch%i' \
                                           % epoch)
                dloader = dloaders[2] if dloaders is not None else None
                plt_utils.plot_predictive_distribution(config, writer, data,
                                    test_struct, mnet, device, dataset,
                                    dloader=dloader, hnet=hnet, figname=figname,
                                    train_iter=train_iter,
                                    publication_style=config.publication_style,
                                    testing_mode='eval')

        logger.info('# Evaluating training ... Done')

def compute_loss(mode, data, mnet, device, config, shared, hnet=None, 
                split_type='test', return_dataset=False,
                return_predictions=False, return_samples=False,
                disable_lrt=False, normal_post=None, reduction='mean',
                dloaders=None):
    r"""Compute the loss over a specified dataset split.

    Note, this method does not execute the code within a ``torch.no_grad()``
    context. This needs to be handled from the outside if desired.
    We expect the networks to be in the correct state (usually the `eval`
    state).

    The number of weight samples evaluated by this method is determined by the
    argument ``config.val_sample_size``, even for the training split!

    The loss computed can be either the MSE (for models with homoscedastic
    noise) or the cross-entropy loss (for models with heteroscedastic noise).
    The derivation is as follows. Overall, we want to minimize the expectation
    over our training samples of the KL divergence between our groundtruth 
    :math:`p(y \mid x)` and our predictions :math:`q(y \mid x, w)`,
    so we want to minimize:

    .. math::

        \mathbb{E}_{p(x)}[D_{KL}(p(y|x) \parallel q(y|x,w))] &= \
            \mathbb{E}_{p(x)}[-H[p(y|x)]] -\mathbb{E}_{p(x,y)}[\log q(y|x,w)] \\
            & \approx \mathbb{E}_{p(x)}[-H[p(y|x)]] \
            -\frac{1}{n}\sum_{i=1}^n\log q(y_i|x_i,w)
    
    where the last approximation is obtained by doing an MC estimate and
    :math:`n` is the number of training samples used.
    We can notice that the first term (the entropy), is constant w.r.t. our
    model :math:`q(y \mid x, w)`, and therefore can be ignored for training.
    The goal thus mecomes to minimize the second term, which is just the
    expectation over :math:`p(x, y)` of the cross-entropy term.

    .. math::

        -\frac{1}{n}\sum_{i=1}^n\log q(y_i|x_i,w) = \frac{1}{n}\sum_{i=1}^n \
            \left(\log (\sigma(x_i)) + \
            \frac{1}{2}\left(\frac{y_i-\mu(x_i)}{\sigma(x_i)}\right)^2 \right) \
            + \text{const}
    
    Throwing away the constant, we end up with the desired loss function:

    .. math::

        L(y, \mu(x), \sigma(x)) = \frac{1}{n}\sum_{i=1}^n\left(\log \
            (\sigma(x_i)) + \frac{1}{2}\left(\frac{y_i-\mu(x_i)}{\
            \sigma(x_i)}\right)^2 \right)

    **Remarks on the MSE value**

    Ideally, we would like to compute the MSE by sampling form the predictive
    posterior for a test sample :math:`(x, y)` as follows

    .. math::

        \mathbb{E}_{\hat{y} \sim p(y \mid \mathcal{D}; x)} \
            \big[ (\hat{y} - y)^2 \big]

    However, since we don't want to sample from the Gaussian likelihood, we
    look at two simplifications of the MSE that only require the evaluation of
    the mean of the likelihood.

    In the first one, we compute the MSE per sampled model and average over
    these MSEs and in the second, we compute the mean over the likelihood means
    and compute the MSE using these "mean" predictions. The relation between
    these two can be shown to be as follows:

    .. math::
        
         \mathbb{E}_{p(W \mid \mathcal{D})} \big[ (f(x, w) - y)^2 \big] = \
             \text{Var}\big( f(x, w) \big) + \
             \Big( \mathbb{E}_{p(W \mid \mathcal{D})} \big[ f(x, w) \big] - \
             y\Big)^2 

    We prefer the first method as it respects the variance of the predictive
    distribution. If we would like to use the MSE as a measure for
    hyperparameter selection, then the model that leads to lower in-distribution
    uncertainty should be preferred (to be precise, the MSE would then have to
    be computed on the training data).

    Args:
        (....): See docstring of method :func:`train`.
        split_type: The name of the dataset split that should be used:

            - ``'test'``: The test set will be used.
            - ``'val'``: The validation set will be used. If not available, the
              test set will be used.
            - ``'train'``: The training set will be used.

        return_dataset: If ``True``, the attributes ``inputs`` and ``targets``
            will be added to the ``return_vals`` Namespace (see return values).
            Both fields will be filled with numpy arrays.

            Note:

                The ``inputs`` and ``targets`` are returned as they are stored
                in the ``data`` dataset handler. I.e., for 1D regression data
                the shapes would be ``[num_samples, 1]``.

        return_predictions: If ``True``, the attribute ``predictions`` will be
            added to the ``return_vals`` Namespace (see return values). These
            fields will correspond to main net outputs for each sample. The
            field will be filled with a numpy array.
        return_samples: If ``True``, the attribute ``samples`` will be added
            to the ``return_vals`` Namespace (see return values). This field
            will contain all weight samples used.
            The field will be filled with a numpy array.
        disable_lrt (bool): Disable the local-reparametrization trick in the
            forward pass of the main network (if it uses it).
        normal_post (tuple, optional): A tuple of lists. The lists have the
            length of the parameter list of the main network. The first list
            represents mean and the second stds of a normal posterior
            distribution. If provided, weights are sampled from this
            distribution. Only implemented for Gaussian likelihood models.
        reduction (str): Whether the NLL loss should be summed ``'sum'`` or
            meaned ``'mean'`` across the batch dimension.

    Returns:
        (tuple): Tuple containing:

        - ``loss``: The mean over the return value ``loss_vals``.
        - ``return_vals``: A namespace object that contains several attributes,
          depending on the arguments passed. It will always contains:

            - ``w_mean``: ``None`` if the hypernetwork encodes an implicit
              distribution. Otherwise, the current mean values of all synapses
              in the main network.
            - ``w_std``: ``None`` if the hypernetwork encodes an implicit
              distribution. Otherwise, the current standard deviations of all
              synapses in the main network or ``None`` if the main network has
              deterministic weights.
            - ``w_hnet``: The output of the hyper-hypernetwork ``hhnet`` for the
              current task, if applicable. ``None`` otherwise.
            - ``loss_vals``: Numpy array of loss values per weight sample.
    """
    assert mode in ['gl', 'ghl', 'nf']
    assert mode != 'nf' or hnet is not None
    assert split_type in ['test', 'val', 'train']
    
    # assume ndims = 2 for BivariateToyRegression datasets 
    # and ndims = 1 for all others
    is_bivariate = isinstance(data, BivariateToyRegression)
    ndims = int(is_bivariate) + 1 
    if dloaders is not None:
        return_samples = False

    return_vals = Namespace()

    # If the hypernetwork is provided, we are in a NF experiment, and the means
    # and variances refer to it.
    net = mnet
    if hnet is not None:
        net = hnet

    # Detect whether we are in a Bayesian or deterministic setting.
    gauss_net = False
    if isinstance(net, GaussianBNNWrapper):
        gauss_net = True

    # Extract model parameters.
    return_vals.w_mean = None
    return_vals.w_std = None
    if gauss_net:
        w_mean, w_rho = net.extract_mean_and_rho()
        w_std = putils.decode_diag_gauss(w_rho, logvar_enc=net.logvar_encoding)
        return_vals.w_mean = w_mean
        return_vals.w_std = w_std

    # Extract the dataloader.
    if dloaders is not None:
        train_dloader, val_dloader, test_dloader = dloaders
        if split_type == 'train':
            dloader = train_dloader
        elif split_type == 'test' or data.num_val_samples == 0:
            dloader = test_dloader
        else:
            dloader = val_dloader
        num_data_points = len(dloader.dataset.imgs)
    else:
        if split_type == 'train':
            X = data.get_train_inputs()
            T = data.get_train_outputs()
        elif split_type == 'test' or data.num_val_samples == 0:
            X = data.get_test_inputs()
            T = data.get_test_outputs()
        else:
            X = data.get_val_inputs()
            T = data.get_val_outputs()
        X = data.input_to_torch_tensor(X, device)
        T = data.output_to_torch_tensor(T, device)
        num_data_points = X.shape[0]
        dloader = [(X, T)]

    # Prepare tensors for storing results.
    num_models = config.val_sample_size
    if return_samples:
        num_w = net.num_params
        if gauss_net:
            num_w = num_w // 2
        return_vals.samples = torch.zeros((num_models, num_w)).to(device)
    loss_vals = torch.zeros(config.val_sample_size).to(device)
    if mode == 'gl':
        mse_vals = torch.zeros(config.val_sample_size).to(device)

    if return_predictions:
        if is_bivariate:
            # 2 mean units, 2 variances and 1 correlation factor.
            predictions = torch.zeros((num_data_points, num_models,2))\
                .to(device)
            predictions_cov = torch.zeros((num_data_points, num_models,3))\
                .to(device)
        else:
            predictions = torch.zeros((num_data_points, num_models)).to(device)
            predictions_std = torch.zeros((num_data_points, num_models))\
                .to(device)
    # For large image datasets we need to iterate. For Toy Datasets, this loop
    # will only run once (i.e. dloader has length one).
    n_seen_inputs = 0
    for i, (X, T) in enumerate(dloader):

        # Move to correct device for steering angle dataset.
        if dloaders is not None:
            X, T = X.to(device), T.to(device)
        # Iterate over the number of models.
        for j in range(num_models):

            # Forward pass.
            if mode == 'nf':
                if normal_post is not None:
                    # Sample weights from Gaussian posterior
                    raise NotImplementedError
                elif gauss_net:
                    flow_weights = hnet.forward(X, weights=None,
                                    mean_only=False, extracted_mean=w_mean,
                                    extracted_rho=w_rho,
                                    disable_lrt=disable_lrt,
                                    ret_format='flattened')
                    Y, log_det = mnet.forward(T, weights=flow_weights)
                else:
                    flow_weights = hnet.forward(X, ret_format='flattened')
                    Y, log_det = mnet.forward(T, weights=flow_weights)

            elif mode == 'gl' or mode == 'ghl':
                weights = None
                if normal_post is not None:
                    # Sample weights from Gaussian posterior
                    weights = []
                    for ii, pmean in enumerate(normal_post[0]):
                        pstd = normal_post[1][ii]
                        weights.append(torch.normal(pmean, pstd,generator=None))
                    Y = mnet.forward(X, weights=weights)
                elif gauss_net:
                    Y = mnet.forward(X, weights=None, mean_only=False,
                                     extracted_mean=w_mean, extracted_rho=w_rho,
                                     disable_lrt=disable_lrt)
                else:
                    Y = mnet.forward(X)
            # Loss computation.
            if mode == 'gl' or mode == 'ghl':
                sigma = config.pred_dist_std if mode == 'gl' else None
                loss_val = compute_nll(T, Y, sigma=sigma, ndims=ndims)
                if mode == 'gl':
                    mse = compute_mse(T, Y, ndims=ndims)
            elif mode == 'nf':
                loss_val = compute_nf_loss(mnet, Y, log_det, reduction='sum' )
            loss_vals[j] += loss_val
            if mode == 'gl':
                # Multiply by batch size in case sizes are different.
                mse_vals[j] += mse * X.shape[0]

            if return_predictions:
                if mode == 'gl' or mode == 'nf':
                    predictions[n_seen_inputs:n_seen_inputs+X.shape[0], j] = \
                                Y.squeeze()
                elif mode == 'ghl':
                    if is_bivariate:
                        predictions[
                            n_seen_inputs:n_seen_inputs+X.shape[0], j] = \
                                Y[:, :2].squeeze()
                        predictions_var = torch.exp(Y[:, 2:4])
                        predictions_rho = torch.tanh(Y[:, 4])[:,np.newaxis]
                        predictions_cov[
                            n_seen_inputs:n_seen_inputs+X.shape[0], j] = \
                                torch.hstack((predictions_var,predictions_rho))

                    else:
                        predictions[
                            n_seen_inputs:n_seen_inputs+X.shape[0], j] = \
                                Y[:, 0].squeeze()
                        predictions_std[
                            n_seen_inputs:n_seen_inputs+X.shape[0], j] = \
                                torch.exp(0.5 * Y[:, 1].squeeze())

            if return_samples:
                # Since this is only called for toy datasets, and there is no
                # loop over the data, we don't need to index according to the
                # number of seen inputs.
                if weights is None:
                    return_vals.samples = None
                else:
                    return_vals.samples[j, :] = torch.cat([p.detach().flatten()\
                        for p in weights])
                return_vals.samples = return_vals.samples.cpu().numpy()

        n_seen_inputs += X.shape[0]

    if return_dataset:
        # For large datasets, only return last batch.
        return_vals.inputs = X
        return_vals.targets = T

    if return_predictions:
        return_vals.predictions = predictions.cpu().numpy()
        if mode == 'ghl':
            if is_bivariate:
                return_vals.predictions_cov = predictions_cov.cpu().numpy()
            else:
                return_vals.predictions_std = predictions_std.cpu().numpy()

    ### Properly normalize the losses according to batch size and models.
    assert n_seen_inputs == num_data_points
    if mode == 'gl':
        mse_vals = mse_vals.detach().cpu().numpy()
        # The MSE is always averaged across the mini-batch. But because it can
        # be that different mini-batches have different sizes, we multiplied
        # above the average mini-batch value by the size of the mini-batch and
        # therefore need to renormalize by the total number of points here.
        mse_vals /= num_data_points
        return_vals.mse_vals = mse_vals

    ## Note that here we should have that:
    # mse_vals * (1/(2 * sigma**2) + (np.log(sigma) + \\
    # 0.5 * np.log(2 * np.pi)) * X.shape[0] == loss_vals

    if reduction == 'mean':
        loss_vals /= num_data_points
    avg_loss_val = loss_vals.mean() # average across models
    loss_vals = loss_vals.detach().cpu().numpy()
    return_vals.loss_vals = loss_vals

    return avg_loss_val, return_vals

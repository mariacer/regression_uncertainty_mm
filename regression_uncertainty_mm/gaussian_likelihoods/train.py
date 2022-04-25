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
# @title          :gaussian_likelihoods/train.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :16/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Training script for Gaussian likelihood regression models
---------------------------------------------------------

This script trains an MLP or BNN-MLP on a regression task.
The likelihood of the data is modeled as a Gaussian distribution with
homoscedastic noise (only the means of the Gaussians are outputted), or 
heteroscedastic noise (the mean and variance of the Gaussian are outputted).
"""
from data.regression2d_heteroscedastic import BivariateHeteroscedasticRegression
import hypnettorch.utils.torch_utils as tutils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from data import data_utils
from gaussian_likelihoods import train_utils
from networks import network_utils as net_utils
from networks.gauss_mnet_interface import GaussianBNNWrapper
from probabilistic import prob_utils as putils
import utils.train_utils as st_utils

def train(data, mnet, device, config, shared, logger, writer, mode, dataset,
          hnet=None, dloaders=None):
    """Train the network.

    The task specific loss aims to learn the mean and variances of the main net
    weights such that the posterior parameter distribution is approximated.

    Args:
        data: The dataset handler.
        mnet: The model of the main network.
        device: Torch device (cpu or gpu).
        config: The command line arguments.
        shared: Miscellaneous data shared among training functions.
        logger: Command-line logger.
        writer: The tensorboard summary writer.
        mode (str): The mode of the experiment.
        dataset (str): The dataset being used.
        hnet (None): Dummy input for compatibility with NFs.
        dloaders (tuple): The training, validation and test dataloaders, only
            provided for the steering angle prediction.

    Returns:
        (bool): Whether the training finished. If a NaN test loss is encountered,
            training is interrupted.
    """
    assert isinstance(mnet, GaussianBNNWrapper) or config.mean_only

    logger.info('Training network...')

    mnet.train()

    # The output space is bivariate for the toy_2D datasets.
    # We assume all other datasets have a monovariate output space.
    is_bivariate = dataset == 'toy_2D'
    ndims = int(is_bivariate)+1

    ### Define Prior
    standard_prior = False
    if config.prior_variance == 1:
        # Use standard Gaussian prior with 0 mean and unit variance.
        standard_prior = True

    params = mnet.parameters()
    optimizer = tutils.get_optimizer(params, config.lr, momentum=None,
        weight_decay=config.weight_decay, use_adam=True,
        adam_beta1=config.adam_beta1)

    if hasattr(config, 'epochs') and config.epochs != -1:
        # If number of epochs is given, it determines the amount of training.
        epochs = config.epochs
        n_iter = int(np.ceil(data.num_train_samples / config.batch_size))
    else:
        # If number of epochs is -1, amount of training is determined by n_iter.
        epochs = 1
        n_iter = config.n_iter
    total_n_iter = n_iter * epochs

    # Unpack dataloaders for steering angle prediction.
    # Note that train loader will be iterated on in the loop below, so it has
    # to have length equal to the number of iterations and consist of tuples,
    # to be usable for non-steering angle prediction tasks.
    train_loader = [(None, None)] * n_iter
    val_loader, test_loader = None, None
    if dloaders is not None:
        train_loader, val_loader, test_loader = dloaders

    interrupted_training = False
    curr_iters = 0
    for e in range(epochs):
        logger.info('')
        logger.info('Training epoch: %d.' % e)

        # Save all predictions for layer analysis.
        all_predictions = []
        for i, (imgs, lbls) in enumerate(train_loader):

            # If we have reached the maximum number of iterations. Will only
            # work if the number of epochs in the command line was -1.
            if curr_iters >= total_n_iter:
                break

            ### Evaluate network.
            # We test the network before we run the training iteration.
            # That way, we can see initial performance of the untrained network.
            if i % config.val_iter == 0:
                st_utils.evaluate(data, mnet, device, config, shared, logger,
                                  writer, mode, dataset, train_iter=curr_iters,
                                  save_fig=True, plot=not config.no_plots,
                                  dloaders=dloaders, epoch=e+1)
                mnet.train()

            if i % 100 == 0:
                logger.info('Training iteration: %d.' % i)

            ### Extract the training data.
            if dloaders is None:
                batch = data.next_train_batch(config.batch_size)
                X = data.input_to_torch_tensor(batch[0], device, mode='train')
                T = data.output_to_torch_tensor(batch[1], device, mode='train')
            else:
                # In the steering angle dataset, since the entire dataset
                # cannot be held in memory, images are loaded batch by batch.
                X, T = imgs.to(device), lbls.to(device)

            ### Train theta and task embedding.
            optimizer.zero_grad()

            if config.mean_only:
                w_mean = mnet.weights
                w_std = None
            else:
                w_mean, w_rho = mnet.extract_mean_and_rho(weights=None)
                w_std, w_logvar = putils.decode_diag_gauss(w_rho, \
                    logvar_enc=mnet.logvar_encoding, return_logvar=True)

            ### Prior-matching loss.
            if config.mean_only:
                loss_kl = 0
            else:
                if standard_prior:
                    # Gaussian prior with zero mean and unit variance.
                    loss_kl = putils.kl_diag_gauss_with_standard_gauss(w_mean,
                        w_logvar)
                else:
                    loss_kl = putils.kl_diag_gaussians(w_mean, w_logvar,
                        shared.prior_mean, shared.prior_logvar)

            ### Compute negative log-likelihood (NLL).
            loss_nll = 0
            mse = 0
            for j in range(config.train_sample_size):
                if config.mean_only:
                    Y = mnet.forward(X, weights=w_mean)
                else:
                    # Note, the sampling will happen inside the forward method.
                    Y = mnet.forward(X, mean_only=False, extracted_mean=w_mean,
                                     extracted_rho=w_rho)
                all_predictions.append(Y)
                # Task-specific loss.
                # We use the reduction method 'mean' on purpose and scale with
                # the number of training samples below.
                sigma = config.pred_dist_std if mode == 'gl' else None
                loss_nll += train_utils.compute_cross_entropy(T, Y, sigma=sigma,
                                                reduction='mean', ndims=ndims)
                # For gl we additionally keep track of the MSE.
                if mode == 'gl':
                    mse += train_utils.compute_mse(T, Y, ndims=ndims)

            # NLL has to be scaled by the dataset size.
            loss_nll *= data.num_train_samples / config.train_sample_size
            loss = loss_kl * config.kl_scale + loss_nll
            if mode == 'gl':
                mse /= config.train_sample_size

            ##  Note that here we have:
            # (mse * (1/(2* sigma**2)) + np.log(sigma) + 0.5*np.log(2*np.pi)) \\
            # * data.num_train_samples == loss_nll

            loss.backward()
            if config.clip_grad_value != -1:
                torch.nn.utils.clip_grad_value_(\
                            optimizer.param_groups[0]['params'],
                            config.clip_grad_value)
            elif config.clip_grad_norm != -1:
                torch.nn.utils.clip_grad_norm_(\
                            optimizer.param_groups[0]['params'],
                            config.clip_grad_norm)
            optimizer.step()

            if i % 50 == 0:
                writer.add_scalar('train/loss', loss, curr_iters)
                if not config.mean_only:
                    writer.add_scalar('train/loss_kl', loss_kl, curr_iters)
                    writer.add_scalar('train/loss_nll',loss_nll, curr_iters)
                if mode == 'gl':
                    writer.add_scalar('train/mse', mse, curr_iters)
                writer.add_scalar('train/num_nans', shared.summary['num_nans'],
                                  curr_iters)

                # Plot distribution of mean and log-variance values.
                mean_outputs = Y[:, 0]
                writer.add_histogram('train/predicted_means', mean_outputs,
                                     curr_iters)
                if mode == 'ghl':
                    std_outputs = torch.exp(1/2 * Y[:, 1])
                    writer.add_histogram('train/predicted_stds', std_outputs,
                                         curr_iters)

            curr_iters += 1
            shared.current_loss = loss
            if mode == 'gl':
                shared.summary['aab_train_mse_final'] = mse

        ### Test the network.
        loss = st_utils.test(data, mnet, device, config, shared, logger, writer,
                    mode, dataset, hnet=hnet, plot=not config.no_plots,
                    dloaders=dloaders, epoch=e+1)
        logger.info('Final loss value after epoch %d: %s' % (e, \
              np.array2string(loss.cpu(), precision=5, separator=',')))

        ### Checkpoint the model at the end of each epoch.
        # Since score is expected to be the higher the better, we provide the 
        # negative loss.
        if config.store_models:
            logger.info('Checkpointing models ...')
            net_utils.save_ckpts(config, -loss, mnet=mnet, iter=e+1,
                    infos={'epoch': e+1, 'test loss': loss,
                           'train loss': shared.current_loss})

        ### For steering angle task, test on splitting lane.
        if dataset == 'steering_angle':
            data_utils.test_splitting_image(mnet, data, device, config, shared,
                                            logger, writer, mode, hnet=hnet,
                                            epoch=e)

        if curr_iters >= total_n_iter:
            break
        plt.close('all')

        if np.isnan(loss.cpu()):
            interrupted_training = True
            break

    logger.info('Training network... Done')

    return not interrupted_training

def run(mode='gl', dataset='toy_1D'):
    st_utils.run(mode=mode, dataset=dataset)
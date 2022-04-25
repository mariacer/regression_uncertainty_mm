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
# @title          :data/plotting_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for plotting different datasets
------------------------------------------------

A collection of helper functions for plotting to keep other scripts clean.
"""
from argparse import Namespace
from hypnettorch.data.special.regression1d_data import ToyRegression
from hypnettorch.data.special.regression1d_bimodal_data import \
        BimodalToyRegression
import hypnettorch.utils.misc as utils
from hypnettorch.data.udacity_ch2 import UdacityCh2Data
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import os
import pickle
from sklearn.neighbors import KernelDensity
import torch
import warnings

from data.regression2d_gaussian import BivariateToyRegression
from data.regression2d_bimodal import BivariateBimodalRegression
from data.regression2d_heteroscedastic import BivariateHeteroscedasticRegression
from data.data_utils import _ONE_D_MISC, _TWO_D_MISC
from data.special.regression1d_data import ToyRegression
from data.special.regression1d_bimodal_data import BimodalToyRegression
from gaussian_likelihoods.train_utils import compute_entropy_gaussian
from normalizing_flows.train_utils import compute_entropy_nf
from utils import math_utils as mutils

_DEFAULT_PLOT_CONFIG = [12, 5, 8] # fontsize, linewidth, markersize
_PUBLICATION_PLOT_CONFIG = [55, 15, 10]

def plot_predictive_distribution(config, writer, data, struct, mnet, device,
                                 dataset, hnet=None, figname=None,
                                 publication_style=False, plot_groundtruth=True,
                                 plot_scatter_dots=True, dloader=None,
                                 plot_continuous_preds=True, train_iter=None,
                                 testing_mode=None, log_likelihood=False):
    """Plot the predictive distribution and groundtruth function.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        data: A data loader.
        struct (Namespace): The structure with input data and predictions.
        mnet: The network.
        device: The cuda device.
        dataset (str): The dataset being used.
        hnet: The (optional) hypernetwork. Provided when the model is a NF.
        figname: The name of the figure. If ``None``, the figure will not be
            saved.
        publication_style: whether plots should be made in publication style.
        plot_groundtruth (boolean, optional): If ``True``, the groundtruth
            function will be plotted.
        plot_scatter_dots (boolean, optional): If ``True``, the dots of the
            training/predictions will be scattered.
        plot_continuous_preds (boolean, optional): If ``True``, the densities
            will be plotted in a continuous space (by considering a grid).
            In this setting, predictions can be averaged or not. If ``False``,
            the plotted predictions will always be averaged.
        train_iter (int): The number of past training iterations. If None, then
            writer will not be used.
        log_likelihood (bool): Whether to plot the likelihood in a log scale.
        dloader: The dataloader (if steering task).
    """
    if config.publication_style:
        plt.rcParams["font.family"] = "Times New Roman"

    if dataset == 'toy_1D':
        plot_pred_dist_1D_dataset(config, writer, data, struct, mnet, device,
                                 hnet=hnet, figname=figname,
                                 publication_style=publication_style,
                                 plot_groundtruth=plot_groundtruth,
                                 plot_scatter_dots=plot_scatter_dots,
                                 plot_continuous_preds=plot_continuous_preds,
                                 train_iter=train_iter,
                                 testing_mode=testing_mode,
                                 log_likelihood=log_likelihood)
    elif dataset =='toy_2D':
        plot_pred_dist_2D_dataset(config, writer, data,
                                 struct, mnet, device, hnet=hnet, 
                                 figname=figname, train_iter=train_iter,
                                 publication_style=publication_style,
                                 testing_mode=testing_mode, 
                                 plot_groundtruth=plot_groundtruth,
                                 plot_scatter_dots=plot_scatter_dots,
                                 plot_continuous_preds=plot_continuous_preds, 
                                 log_likelihood=log_likelihood)
    elif dataset == 'steering_angle':
        plot_pred_dist_steering(config, writer, data, mnet, device,
                                 dloader=dloader, hnet=hnet, figname=figname,
                                 publication_style=publication_style,
                                 plot_groundtruth=plot_groundtruth,
                                 plot_scatter_dots=plot_scatter_dots,
                                 train_iter=train_iter,
                                 testing_mode=testing_mode,
                                 log_likelihood=log_likelihood,
                                 plot_images=config.plot_steering_images)

    if testing_mode is not None:
        testing_mode = testing_mode + '/'
    else:
        testing_mode = ''
    writer.add_figure(testing_mode + 'test_predictions', plt.gcf(),
                      train_iter, close=not config.show_plots)

    if config.show_plots:
        utils.repair_canvas_and_show_fig(plt.gcf())

def plot_pred_dist_1D_dataset(config, writer, data, struct, mnet, device,
                              hnet=None, figname=None,
                              publication_style=False, plot_groundtruth=True,
                              plot_scatter_dots=True,
                              plot_continuous_preds=True, train_iter=None,
                              testing_mode=None, log_likelihood=False):
    """Plot predictive distribution for the 1D toy dataset.

    Args:
        (....): See docstring of function :func:`plot_predictive_distribution`.
    """
    if config.publication_style and config.num_train >500:
        plot_scatter_dots = False

    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.publication_style:
        ts, lw, ms = _PUBLICATION_PLOT_CONFIG

    assert isinstance(data, ToyRegression) or \
        isinstance(data, BimodalToyRegression)
    # Not continuous predictions can only be done for 'gl' and 'ghl'.
    assert plot_continuous_preds == True or hnet is None

    ### Set some matplotlib parameters.
    colors = utils.get_colorbrewer2_colors(family='Dark2')
    fig, axes = plt.subplots(figsize=(11, 8))

    ### Define plotting range.
    train_range = data.train_x_range
    range_offset = (train_range[1] - train_range[0]) * 0.05

    ### Plot the groundtruth function.
    if plot_groundtruth:
        sample_x, sample_y = data._get_function_vals( \
            x_range=[train_range[0]-range_offset, train_range[1]+range_offset])

        # Plot the groundtruth function.
        if isinstance(data, BimodalToyRegression):
            plt.plot(sample_x, sample_y-data._dist1, color='k',
                     linestyle='dashed', linewidth=lw/8.)
            plt.plot(sample_x, sample_y+data._dist2, color='k',
                     linestyle='dashed', linewidth=lw/8., label='Function')
        elif isinstance(data, ToyRegression):
            plt.plot(sample_x, sample_y, color='k', linestyle='dashed',
                     linewidth=lw/8., label='Function')

        if plot_scatter_dots:
            train_x = data.get_train_inputs().squeeze()
            train_y = data.get_train_outputs().squeeze()
            plt.plot(train_x, train_y, 'o', color='k', label='Training Data',
                markersize=ms*0.8)

    ### Plot the predictions.
    inputs = struct.inputs.squeeze()
    if plot_continuous_preds:
        x, x_vis, y, grid_size = prepare_grids(_ONE_D_MISC)

        ### Make predictions on the entire x-range.
        X = data.input_to_torch_tensor(x, device)
        if hnet is not None:
            # Need a matrix with all y values at which to evaluate the density.
            T = data.input_to_torch_tensor(y, device)
            T = torch.vstack([T] * grid_size[0])
            T = torch.transpose(T, 0, 1)
            T = torch.reshape(T, (-1,))

        if hnet is None:
            predictions = np.zeros((grid_size[0], config.val_sample_size))
            predictions_std = np.ones((grid_size[0], config.val_sample_size)) \
                * config.pred_dist_std # default std, overwritten below for ghl
        else:
            # FIXME: if grid is not square, this will fail: 
            # shape should be (grid_size[1],grid_size[0],config.val_sample_size)
            predictions = np.zeros((*grid_size, config.val_sample_size))

        for j in range(config.val_sample_size):
            if hnet is None: # gl or ghl
                if config.mean_only:
                    Y = mnet.forward(X)
                else:
                    w_mean, w_rho = mnet.extract_mean_and_rho()
                    Y = mnet.forward(X, weights=None, mean_only=False,
                                     extracted_mean=w_mean, extracted_rho=w_rho,
                                     disable_lrt=False)
                predictions[:, j] = Y[:, 0].detach().cpu().numpy().squeeze()
                if log_likelihood:
                    predictions[:, j] = np.log(predictions[:, j])
                if Y.shape[1] == 2:
                    rho = Y[:, 1].detach().cpu().numpy().squeeze()
                    predictions_std[:, j] = np.exp(0.5 * rho)   
            else:
                if config.mean_only:
                    flow_weights = hnet.forward(X, ret_format='flattened')
                else:
                    w_mean, w_rho = hnet.extract_mean_and_rho(weights=None)
                    # Note, the sampling will happen inside the forward method.
                    flow_weights = hnet.forward(X, weights=None, mean_only=False,
                                                extracted_mean=w_mean,
                                                extracted_rho=w_rho,
                                                ret_format='flattened')
                flow_weights = flow_weights.repeat(grid_size[1], 1)
                Y, logabsdet = mnet.forward(T, weights=flow_weights)
                Y = mnet.log_p0(Y) + logabsdet
                if not log_likelihood:
                    Y = torch.exp(Y)
                Y = torch.reshape(Y, (grid_size[1], grid_size[0]))
                predictions[:, :, j] = Y.detach().cpu().numpy().squeeze()

        ### For each x-value, determine the density, averaged across models.
        z = np.zeros((grid_size[1], grid_size[0]))
        for j in range(config.val_sample_size):
            for i in range(grid_size[0]):
                if hnet is None: # gl or ghl
                    z[:, i] += mutils.normal_dist(y, predictions[i, j], \
                                                  predictions_std[i, j])
                else:
                    z[:, i] += predictions[:, i, j]
        z /= config.val_sample_size

        # Crop z to max in-distribution value.
        ind_idxs = np.where(np.abs(x) < 4)[0] # in-distribution indexes
        max_indistribution = z[:, ind_idxs].max() # max in-distribution value
        z[z > max_indistribution] = max_indistribution

        ### Plot.
        args = [z]
        kwargs = {'cmap': 'Blues', 'aspect': 'auto', 'origin': 'lower',
                  'extent': (x_vis[0], x_vis[-1], y[0], y[-1]),
                  'interpolation': 'bicubic'}
        f = plt.imshow(*args, **kwargs)
        cbar = plt.colorbar(f)
        if config.publication_style:
            cbar.ax.tick_params(labelsize=ts, length=lw, width=lw/2.)

    else:
        # Plot the distributions by looking at the models one by one, and
        # using the preset variance (gl) or the outputted variance (ghl).
        # Only applicable to GL or GHL.
        assert hnet is None

        for j in range(config.val_sample_size):
            preds_mean = struct.predictions[:, j]
            if hasattr(struct, 'predictions_std'):
                # Heteroscedastic predictions.
                preds_std = struct.predictions_std[:, j]            
            else: 
                # Homoscedastic predictions.
                preds_std = np.ones_like(preds_mean) * config.pred_dist_std
            plt.plot(inputs, preds_mean, color=colors[0], lw=lw/3.)
            plt.fill_between(inputs, preds_mean - preds_std,
                                     preds_mean + preds_std, 
                             alpha=0.1, color=colors[0])

        if plot_scatter_dots:
            for i in range(config.val_sample_size):
                plt.plot(inputs, struct.predictions[:, i], 'o', color=colors[0],
                    label='Predictions' if i == 0 else None, markersize=ms)

    if plot_groundtruth:
        # Plot training data boundaries.
        for ii, x_lim in enumerate(train_range):
            plt.plot([x_lim, x_lim], [plt.ylim()[0], plt.ylim()[1]], color='k',
                    linestyle='dotted', label='Training boundary' if ii == 0
                    else None, linewidth=lw/12.)

    title = 'Predictive distribution'
    if log_likelihood:
        title += ' (logscale)'

    if publication_style:
        make_1D_for_publication(axes)
        #plt.title(title, fontsize=ts, pad=ts)
        #plt.legend(fontsize=ts)
    else:
        plt.legend()
        plt.title(title, fontsize=ts, pad=ts)
        plt.xlabel('$x$', fontsize=ts)
        plt.ylabel('$y$', fontsize=ts)

    if train_iter is not None:
        figname = figname + '_iter%s' % train_iter
    if figname:
        plt.savefig(figname + '.pdf', bbox_inches='tight')

    ## Plot x-slice.
    # plt.figure()
    # std = np.sqrt(config.gaussian_variance)
    # mode_1 = mutils.normal_dist(y, y[100]**3 + 50, std)
    # mode_2 = mutils.normal_dist(y, y[100]**3 - 50, np.sqrt(std))
    # plt.plot(y, 0.5*mode_1 + 0.5*mode_2)
    # plt.plot(y, z[:, 100], color='r')
    # plt.title('x = %f' % y[100])
    # if figname:
    #     plt.savefig(figname + '_slice.pdf', bbox_inches='tight')

def make_1D_for_publication(axes, entropy=False):
    """Fix some matplotlib options to obtain publication ready plots for 1D.

    Args:
        axes: The current axes.
    """
    ts, lw, ms = _PUBLICATION_PLOT_CONFIG

    axes.grid(False)
    axes.set_facecolor('w')
    axes.set_xlim([-6.5, 6.5])

    # Draw left and lower lines.
    axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
    axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
    if entropy:
        axes.axvline(x=axes.get_xlim()[1], color='k', lw=lw)

    # Get the x ticks labels to have the right fonts.
    for tick in axes.xaxis.get_major_ticks():
        tick.label.set_fontsize(ts)
    for tick in axes.yaxis.get_major_ticks():
        tick.label.set_fontsize(ts)
    axes.set_xticks([-5, 0, 5])

    # Remove y ticks.
    if not entropy:
        axes.tick_params(axis='y', which='both', left=False, labelleft=False)
    else:
        axes.tick_params(axis='y', which='both', left=True, labelleft=True,
                         right=False, labelright=False)
    #axes.set_yticks([-200, 0, 200])

    # Determine the size of the ticks.
    axes.tick_params(axis='both', length=lw, direction='out', width=lw/2.)

    # Remove top and left lines.
    if not entropy:
        axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)


def make_2D_for_publication(axes, plot_marginal=False):
    """Fix some matplotlib options to obtain publication ready plots for 1D.

    Args:
        axes: The current axes.
    """
    if plot_marginal:
        ax, ax_x, ax_y = axes
    ts, lw, ms = _PUBLICATION_PLOT_CONFIG
    
    
    zoom = 15
    xlim = ax.get_xlim()
    ax.set_xlim([xlim[0] + zoom, xlim[1] - zoom])

    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0] + zoom, ylim[1] - zoom])



    if plot_marginal:
        # ax_x.set_yticks([0])

        for a in axes:
            a.grid(False)
            a.set_facecolor('w')
            # Draw left and lower lines.
            a.axhline(y=a.get_ylim()[0], color='k', lw=lw)
            a.axvline(x=a.get_xlim()[0], color='k', lw=lw)
            a.xaxis.set_major_locator(plt.MaxNLocator(1))
            a.yaxis.set_major_locator(plt.MaxNLocator(1))
                # Get the x ticks labels to have the right fonts.
        ax_y.set_xticks([0])
        for a in axes:
            for tick in a.xaxis.get_major_ticks():
                tick.label.set_fontsize(ts)
            for tick in a.yaxis.get_major_ticks():
                tick.label.set_fontsize(ts)

    else:
        ax.grid(False)
        ax.set_facecolor('w')
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # Draw left and lower lines.
        ax.axhline(y=ax.get_ylim()[0], color='k', lw=lw)
        ax.axvline(x=ax.get_xlim()[0], color='k', lw=lw)
        # Get the x ticks labels to have the right fonts.
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(ts)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(ts)

    # Remove y ticks.
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.tick_params(axis='y', which='both', left=True, labelleft=True,
                        right=False, labelright=False)
    #ax.set_yticks([-200, 0, 200])

    # Determine the size of the ticks.
    ax.tick_params(axis='both', length=lw, direction='out', width=lw/2.)

    # Remove top and left lines.
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_pred_dist_steering(config, writer, data, mnet, device,
                            dloader=None, hnet=None, figname=None,
                            publication_style=False, plot_groundtruth=True,
                            plot_scatter_dots=True,
                            plot_continuous_preds=False, train_iter=None,
                            testing_mode=None, log_likelihood=False,
                            input_images=None, plot_images=True,
                            save_predictions=False):
    """Plot predictive distribution for the steering angle prediction.

    Args:
        (....): See docstring of function :func:`plot_predictive_distribution
        input_images (list): An optional set of input images to be plotted.
            If ``None``, a random set will be selected. If this is provided,
            ``data`` may be None.
        plot_images (bool): Whether individual images should be plotted as well.
    """
    assert isinstance(data, UdacityCh2Data)
    # Not continuous predictions can only be done for 'gl' and 'ghl'.
    if hnet is not None:
        plot_continuous_preds = True

    num_images = config.num_plotted_predictions
    if input_images is not None:
        num_images = len(input_images)

    ### Set some matplotlib parameters.
    fig, axes = plt.subplots(figsize=(11, 8))
    colors = utils.get_colorbrewer2_colors(family='Dark2')

    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.publication_style:
        ts, lw, ms = _PUBLICATION_PLOT_CONFIG

    ### Define plotting range.
    train_range = [-np.pi, +np.pi]
    num_y_points = 200
    y = np.linspace(train_range[0], train_range[1], num_y_points)

    # Randomly select images from dataset.
    if input_images is None:
        rstate = np.random.RandomState(config.random_seed)
        img_idxs = rstate.randint(0, len(dloader.dataset.imgs), num_images)
    else:
        img_idxs = np.arange(num_images)

    if train_iter is not None:
        figname = figname + '_iter%s' % train_iter
    for i in img_idxs:

        # Extract image and target.
        if input_images is not None:
            X = input_images[i]
        else:
            X = dloader.dataset[i][0].unsqueeze(dim=0).to(device)
            T = dloader.dataset[i][1]

        fig, axes = plt.subplots(figsize=(12, 6))
        ### Plot the predictions.
        if plot_continuous_preds:

            if hnet is None:
                predictions = np.zeros(config.val_sample_size)
                predictions_std = np.ones(config.val_sample_size) \
                    * config.pred_dist_std # default std, overwritten for ghl
            else:
                # Need a matrix with all values at which to evaluate the density.
                t = torch.tensor(y).to(device)
                predictions = np.zeros((num_y_points, config.val_sample_size))

            for j in range(config.val_sample_size):
                if hnet is None: # gl or ghl
                    if config.mean_only:
                        Y = mnet.forward(X)
                    else:
                        w_mean, w_rho = mnet.extract_mean_and_rho()
                        Y = mnet.forward(X, weights=None, mean_only=False,
                                         extracted_mean=w_mean,
                                         extracted_rho=w_rho,
                                         disable_lrt=False)
                    predictions[j] = Y[:, 0].detach().cpu().numpy().squeeze()
                    if log_likelihood:
                        predictions[j] = np.log(predictions[j])
                    if Y.shape[1] == 2:
                        rho = Y[:, 1].detach().cpu().numpy().squeeze()
                        predictions_std[j] = np.exp(0.5 * rho)   
                else:
                    if config.mean_only:
                        flow_weights = hnet.forward(X, ret_format='flattened')
                    else:
                        w_mean, w_rho = hnet.extract_mean_and_rho(weights=None)
                        # Note, sampling will happen inside the forward method.
                        flow_weights = hnet.forward(X, weights=None,
                                                    mean_only=False,
                                                    extracted_mean=w_mean,
                                                    extracted_rho=w_rho,
                                                    ret_format='flattened')

                flow_weights = flow_weights.repeat(num_y_points, 1)
                Y, logabsdet = mnet.forward(t, weights=flow_weights)
                Y = mnet.log_p0(Y) + logabsdet
                if not log_likelihood:
                    Y = torch.exp(Y)
                predictions[:, j] = Y.detach().cpu().numpy().squeeze()

            ### For each x-value, determine the density, averaged across models.
            z = np.zeros(num_y_points)
            for j in range(config.val_sample_size):
                if hnet is None: # gl or ghl
                    z += mutils.normal_dist(y, predictions[j], \
                                               predictions_std[j])
                else:
                    z += predictions[:, j]
            z /= config.val_sample_size

            ### Plot.
            plt.plot(y, z)

            if save_predictions:
                # Store as pickle.
                predictions_dict = {'z': z,
                                    'range': y}
                name = figname + '_predictions_idx%i.pkl' % (i)
                with open(name,'wb') as f:
                    pickle.dump(predictions_dict, f)
        else:
            # Plot the distributions by looking at the models one by one, and
            # using the preset variance (gl) or the outputted variance (ghl).
            # Only applicable to GL or GHL.
            assert hnet is None

            for j in range(config.val_sample_size):
                if config.mean_only:
                    Y = mnet.forward(X)
                else:
                    w_mean, w_rho = mnet.extract_mean_and_rho()
                    Y = mnet.forward(X, weights=None, mean_only=False,
                                     extracted_mean=w_mean, extracted_rho=w_rho,
                                     disable_lrt=False)

                predictions = Y[:, 0].detach().cpu().numpy().squeeze()
                predictions_std = config.pred_dist_std
                if log_likelihood:
                    predictions = np.log(predictions)
                if Y.shape[1] == 2:
                    rho = Y[:, 1].detach().cpu().numpy().squeeze()
                    predictions_std = np.exp(0.5 * rho)   

                py = mutils.normal_dist(y, predictions, predictions_std)
                plt.plot(y, py, color=colors[0], lw=lw/7.)

                if save_predictions:
                    # Store as pickle.
                    predictions_dict = {'mean': py,
                                        'std': predictions_std,
                                        'range': y}
                    name = figname + '_predictions%i_idx%i.pkl' % (j, i)
                    with open(name,'wb') as f:
                        pickle.dump(predictions_dict, f)

        ### Plot the groundtruth function.
        if plot_groundtruth:
            plt.plot([T, T], axes.get_ylim(), color='k', linestyle='dashed',
                     linewidth=lw/7., label='groundtruth')

        title = 'Predictive distribution'
        if log_likelihood:
            title += ' (logscale)'
        if publication_style:
            axes.grid(False)
            axes.set_facecolor('w')
            axes.set_ylim([-2.5, 3])
            axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
            axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
            for tick in axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(ts)
            for tick in axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(ts)
            axes.tick_params(axis='both', length=lw, direction='out',
                             width=lw/2.)
            plt.title(title, fontsize=ts, pad=ts)
        else:
            if plot_groundtruth:
                plt.legend()
            plt.title(title, fontsize=ts, pad=ts)
        plt.xlabel(r'$\theta$', fontsize=ts)
        plt.ylabel(r'$p(\theta)$', fontsize=ts)

        if figname:
            plt.savefig(figname + '_idx%i.pdf' % i, bbox_inches='tight')
            writer.add_figure(figname + '_idx%i' % i, plt.gcf(),
                              train_iter, close=not config.show_plots)
            if config.show_plots:
                utils.repair_canvas_and_show_fig(plt.gcf())

            if plot_images:
                tmp = np.chararray([1,1],
                                   itemsize=len(dloader.dataset.imgs[i][0]),
                                   unicode=True)
                tmp[0, :] = dloader.dataset.imgs[i][0]
                data.plot_samples('Sample %i' % i, tmp, \
                               np.array([dloader.dataset.imgs[i][1]]),
                               predictions=predictions.reshape(1, 1),
                               filename=figname + '_image_idx%i' % i,
                               show=False)


def plot_entropy(config, writer, data, struct, mnet, device, dataset, hnet=None,
                 figname=None, publication_style=False, train_iter=None,
                 testing_mode=None, num_models=100, num_samples=1000):
    """Plot the differential entropy of the predictive posterior.

    Args:
        (....): See docstring of function :func:`plot_predictive_distribution`.
        num_models (int): The number of models to be used for the MC estimate.
        num_samples (int): The number of samples to be used for the MC estimate.
    """
    assert isinstance(data, ToyRegression) or \
        isinstance(data, BimodalToyRegression)

    ### Set some matplotlib parameters.
    colors = utils.get_colorbrewer2_colors(family='Dark2')
    fig, axes = plt.subplots(figsize=(8.8, 2))

    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.publication_style:
        ts, lw, ms = _PUBLICATION_PLOT_CONFIG

    ### Define plotting range and prepare the grid.
    train_range = data.train_x_range
    range_offset = (train_range[1] - train_range[0]) * 0.05
    x, x_vis, _, grid_size = prepare_grids(_ONE_D_MISC)
    x_torch = data.input_to_torch_tensor(x, device)

    ### Compute the entropy.
    if hnet is None: 
        entropy = compute_entropy_gaussian(mnet, config, x_torch,
                                           num_samples=num_samples,
                                           num_models=num_models)
    else:
        entropy = compute_entropy_nf(mnet, hnet, config, x_torch, grid_size[0],
                                     num_samples=num_samples,
                                     num_models=num_models)

    plt.plot(x, entropy, lw=lw/3., color='darkorange')
    title = 'Differential entropy'
    if publication_style:
        make_1D_for_publication(axes, entropy=True)
        axes.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    else:
        plt.title(title, fontsize=ts, pad=ts)
        plt.ylabel(r'$\mathcal{H}$', fontsize=ts)
        plt.xlabel('$x$', fontsize=ts)

    if train_iter is not None:
        figname = figname + '_iter%s' % train_iter
    if figname:
        plt.savefig(figname + '.pdf', bbox_inches='tight')

    if testing_mode is not None:
        testing_mode = testing_mode + '/'
    else:
        testing_mode = ''
    writer.add_figure(testing_mode + 'entropy', plt.gcf(),
                      train_iter, close=not config.show_plots)

    if config.show_plots:
        utils.repair_canvas_and_show_fig(plt.gcf())


def plot_pred_dist_2D_dataset(config, writer, data, struct, mnet, device,
                                 hnet=None, figname=None,
                                 publication_style=False, plot_groundtruth=True,
                                 plot_scatter_dots=False,
                                 plot_continuous_preds=True, train_iter=None,
                                 testing_mode=None, log_likelihood=False):
    """Plot the predictive distribution and groundtruth function.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        data: A data loader.
        struct (Namespace): The structure with input data and predictions.
        mnet: The network.
        device: The cuda device.
        hnet: The (optional) hypernetwork. Provided when the model is a NF.
        figname: The name of the figure. If ``None``, the figure will not be
            saved.
        publication_style: whether plots should be made in publication style.
        plot_groundtruth (boolean, optional): If ``True``, the groundtruth
            function will be plotted.
        plot_scatter_dots (boolean, optional): If ``True``, the dots of the
            training/predictions will be scattered.
        plot_continuous_preds (boolean, optional): If ``True``, the densities
            will be plotted in a continuous space (by considering a grid).
            In this setting, predictions can be averaged or not. If ``False``,
            the plotted predictions will always be averaged.
        train_iter (int): The number of past training iterations. If None, then
            writer will not be used.
        average_predictions (boolean, optional): If `True`, the predictions
            across different model samples will be averaged, and the plot will
            then show the mean and standard deviation (unimodality assumption).
            If `False`, kernel density estimation will be done to plot the
            density at each x value, thus allowing for multimodality.
    """


    assert isinstance(data, BivariateToyRegression)
    # Not continuous predictions can only be done for 'gl' and 'ghl'.
    assert plot_continuous_preds == True or hnet is None

    # plot_pred_dist_2D_marginalized(config, writer, data, struct, 
    #                         mnet, device, hnet, figname, publication_style, 
    #                         plot_groundtruth, plot_scatter_dots, 
    #                         plot_continuous_preds, train_iter, testing_mode,
    #                         log_likelihood=log_likelihood)
    # plot_pred_dist_2D_conditional(config, writer, data, struct, 
    #                         mnet, device, hnet, figname, publication_style, 
    #                         plot_groundtruth, plot_scatter_dots, 
    #                         plot_continuous_preds, train_iter, testing_mode, 
    #                         x_bar = 0)
    # if isinstance(data,BivariateHeteroscedasticRegression):
        # Plot at different values of x
    train_range = data.train_x_range
    for x_bar in range(train_range[0],train_range[1],2):
        plot_pred_dist_2D_conditional(config, writer, data, struct, 
                    mnet, device, hnet, figname, publication_style, 
                    plot_groundtruth, plot_scatter_dots, 
                    plot_continuous_preds, train_iter, testing_mode, 
                    x_bar = x_bar,log_likelihood=log_likelihood)
            


def plot_pred_dist_2D_marginalized(config, writer, data, struct, 
                                 mnet, device, hnet=None, figname=None, 
                                 publication_style=False, plot_groundtruth=True,
                                 plot_scatter_dots=False,
                                 plot_continuous_preds=True, train_iter=None,
                                 testing_mode=None, log_likelihood=False):
    """Plot the predictive distribution and groundtruth function marginalized 
    over each dimension of the output space.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        data: A data loader.
        struct (Namespace): The structure with input data and predictions.
        mnet: The network.
        device: The cuda device.
        hnet: The (optional) hypernetwork. Provided when the model is a NF.
        figname: The name of the figure. If ``None``, the figure will not be
            saved.
        publication_style: whether plots should be made in publication style.
        plot_groundtruth (boolean, optional): If ``True``, the groundtruth
            function will be plotted.
        plot_scatter_dots (boolean, optional): If ``True``, the dots of the
            training/predictions will be scattered.
        plot_continuous_preds (boolean, optional): If ``True``, the densities
            will be plotted in a continuous space (by considering a grid).
            In this setting, predictions can be averaged or not. If ``False``,
            the plotted predictions will always be averaged.
        train_iter (int): The number of past training iterations. If None, then
            writer will not be used.
        average_predictions (boolean, optional): If `True`, the predictions
            across different model samples will be averaged, and the plot will
            then show the mean and standard deviation (unimodality assumption).
            If `False`, kernel density estimation will be done to plot the
            density at each x value, thus allowing for multimodality.
    """
    assert isinstance(data, BivariateToyRegression)
    # Not continuous predictions can only be done for 'gl' and 'ghl'.
    assert plot_continuous_preds == True or hnet is None


    ### Define plotting range.
    train_range = data.train_x_range
    range_offset = (train_range[1] - train_range[0]) * 0.05

    ### Plot the groundtruth function.
    if plot_groundtruth:
        sample_x, sample_y = data._get_function_vals( \
            x_range=[train_range[0]-range_offset, train_range[1]+range_offset])

        if plot_scatter_dots:
            train_x = data.get_train_inputs().squeeze()
            train_y = data.get_train_outputs()

    ### Get the predictions.
    inputs = struct.inputs.squeeze()
    if plot_continuous_preds:
        # Plot continuous predictions without assuming normality (taking mean).

        x, x_vis, y, grid_size = prepare_grids(_ONE_D_MISC)

        ### Make predictions on the entire x-range.
        predictions = np.empty(
            (x.shape[0], config.val_sample_size, data.out_shape[0]))
        X = data.input_to_torch_tensor(x, device)

        for j in range(config.val_sample_size):
            if hnet is None: # gl or ghl
                if config.mean_only:
                    Y = mnet.forward(X)
                else:
                    w_mean, w_rho = mnet.extract_mean_and_rho()
                    Y = mnet.forward(X, weights=None, mean_only=False,
                                     extracted_mean=w_mean, extracted_rho=w_rho,
                                     disable_lrt=False)
            else:
                # FIXME
                pass
            predictions[:, j, :] = Y[:,:2].detach().numpy()
        
        # if average_predictions:
        mean_pred = predictions.mean(axis=1)
        std_pred = predictions.std(axis=1)
        
        ### For each x-value, determine the density.
        # This will be done either:
        # - based on the computed mean and average (`average_predictions==True`)
        # - by doing KDE (`average_predictions==False`).
        z = np.empty((grid_size[1],grid_size[0],data.out_shape[0]))
        for i in range(grid_size[0]):
            for ax in range(2):
                # if average_predictions:
                z[:, i, ax] = mutils.normal_dist(
                    y, mean_pred[i,ax], std_pred[i, ax])
                # else:
                # z[:, i, ax] = kde_samples(predictions[i, :, ax], y)
    else:
        # if average_predictions:
            # Plot the distributions by averaging the provided outputs
            # (several models sampled), and using the standard deviation
            # across models.
        sort_idx = np.argsort(inputs)
        sorted_inputs = inputs[sort_idx]
        preds_mean = struct.predictions.mean(axis=1)[sort_idx]
        preds_std = struct.predictions.std(axis=1)[sort_idx]
    
    in_figname = figname
    for axis in range(2):
        ### Set some matplotlib parameters.
        colors = utils.get_colorbrewer2_colors(family='Dark2')
        fig, axes = plt.subplots(figsize=(12, 6))

        ts, lw, ms = _DEFAULT_PLOT_CONFIG
        if config.publication_style:
            ts, lw, ms = _PUBLICATION_PLOT_CONFIG

        # The default matplotlib setting is usually too high for most plots.
        plt.locator_params(axis='y', nbins=2)
        plt.locator_params(axis='x', nbins=6)
            
        ### Plot the groundtruth function.
        if plot_groundtruth:
            # Plot the groundtruth function marginalized.
            if isinstance(data, BivariateBimodalRegression):
                plt.plot(sample_x, (sample_y+data.mean_mode1)[:,axis], color='k',
                        linestyle='dashed', linewidth=lw/7.)
                plt.plot(sample_x, (sample_y+data.mean_mode2)[:,axis], color='k',
                        linestyle='dashed', linewidth=lw/7., label='Function')
            elif isinstance(data, BivariateToyRegression):
                plt.plot(sample_x, sample_y[:,axis], color='k', linestyle='dashed',
                        linewidth=lw/7., label='Function')
        if plot_scatter_dots:
            plt.plot(train_x, train_y[:,axis], 'o', color='k', label='Training Data',
                markersize=ms)
        if plot_continuous_preds:
                    ### Plot.
            args = [z[..., axis]]
            kwargs = {'cmap': 'Blues', 'aspect': 'auto', 'origin': 'lower',
                    'extent': (x_vis[0], x_vis[-1], y[0], y[-1]),
                    'interpolation': 'bicubic'}
            f = plt.imshow(*args, **kwargs)
            plt.colorbar(f)
        else:
            # if average_predictions:
            plt.plot(inputs, preds_mean[:,axis], color=colors[0], lw=lw/7.)
            plt.fill_between(inputs, preds_mean[:,axis] - preds_std[:,axis],
                            preds_mean[:,axis] + preds_std[:,axis], 
                            alpha=0.2, color=colors[0])

        
        if plot_groundtruth:
            # Plot training data boundaries.
            for ii, x_lim in enumerate(train_range):
                plt.plot([x_lim, x_lim], [plt.ylim()[0], plt.ylim()[1]], color='k',
                        linestyle='dotted', label='Training boundary' if ii == 0
                        else None)

        if publication_style:
            axes.grid(False)
            axes.set_facecolor('w')
            axes.set_ylim([-2.5, 3])
            axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
            axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
            for tick in axes.yaxis.get_major_ticks():
                tick.label.set_fontsize(ts)
            for tick in axes.xaxis.get_major_ticks():
                tick.label.set_fontsize(ts)
            axes.tick_params(axis='both', length=lw/3., direction='out', width=lw/7.)
            axes.set_yticks([-200, 0, 200])
            axes.set_xticks([-5, 0, 5])
            plt.title('Marginal predictive distribution ', fontsize=ts, pad=ts)
        else:
            plt.legend()
            plt.title('Marginal predictive distribution')
        plt.xlabel('$x$', fontsize=ts)
        plt.ylabel(f'$y_{axis}$', fontsize=ts)

        figname = f'{in_figname}_y{axis}'

        if train_iter is not None:
            figname = figname + '_%s' % train_iter
        if figname:
            plt.savefig(figname + '.pdf', bbox_inches='tight')

        if testing_mode is not None:
            testing_mode = testing_mode + '/'
        sfx = ''
        writer.add_figure(testing_mode + 'test_predictions' + sfx, plt.gcf(),
                        train_iter, close=not config.show_plots)

        if config.show_plots:
            utils.repair_canvas_and_show_fig(plt.gcf())
        
def plot_pred_dist_2D_conditional(config, writer, data, struct, 
                                 mnet, device, hnet=None, figname=None, 
                                 publication_style=False, plot_groundtruth=True,
                                 plot_scatter_dots=False,
                                 plot_continuous_preds=True, train_iter=None,
                                 testing_mode=None, log_likelihood=False, 
                                 plot_marginal=True, x_bar = 0):
    """Plot the predictive distribution and groundtruth function marginalized 
    over each dimension of the output space.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        data: A data loader.
        struct (Namespace): The structure with input data and predictions.
        mnet: The network.
        device: The cuda device.
        hnet: The (optional) hypernetwork. Provided when the model is a NF.
        figname: The name of the figure. If ``None``, the figure will not be
            saved.
        publication_style: whether plots should be made in publication style.
        plot_groundtruth (boolean, optional): If ``True``, the groundtruth
            function will be plotted.
        plot_scatter_dots (boolean, optional): If ``True``, the dots of the
            training/predictions will be scattered.
        plot_continuous_preds (boolean, optional): If ``True``, the densities
            will be plotted in a continuous space (by considering a grid).
            In this setting, predictions can be averaged or not. If ``False``,
            the plotted predictions will always be averaged.
        train_iter (int): The number of past training iterations. If None, then
            writer will not be used.
        average_predictions (boolean, optional): If `True`, the predictions
            across different model samples will be averaged, and the plot will
            then show the mean and standard deviation (unimodality assumption).
            If `False`, kernel density estimation will be done to plot the
            density at each x value, thus allowing for multimodality.
    """
    ### Set some matplotlib parameters.
    colors = utils.get_colorbrewer2_colors(family='Dark2')
    if plot_marginal:
        fig, ax, ax_x, ax_y = setup_conditional_fig(figsize=(11,9.7),width=0.6)
    else:
        ### Set some matplotlib parameters.
        fig, ax = plt.subplots(figsize=(8, 7))
        # The default matplotlib setting is usually too high for most plots.
        plt.locator_params(axis='y', nbins=3)
        plt.locator_params(axis='x', nbins=3)


    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.publication_style:
        ts, lw, ms = _PUBLICATION_PLOT_CONFIG


    # sample_x, sample_y = data._get_function_vals( \
    # x_range=[train_range[0]-range_offset, train_range[1]+range_offset])
    sample_y = data._map(x_bar)
    # Plot the groundtruth contour distribution.
    if isinstance(data,BivariateHeteroscedasticRegression):
        cov = data._get_cov_matrix(x_bar)
    else:
        cov = data.cov_matrix
    radius_x,radius_y,angle = get_ellipse_axis(cov)
    if plot_groundtruth:
        if isinstance(data, BivariateBimodalRegression):
            pos1 = sample_y+data._mean1
            ax.plot(*pos1, 'k+',label='Function mean')
            ellipse1 = Ellipse(pos1, width=radius_x*2, height=radius_y*2,
                                angle=np.degrees(angle),facecolor='None',
                                edgecolor='k',linestyle='--',
                                label='True Noise contour')
            ax.add_patch(ellipse1)

            pos2 = sample_y+data._mean2
            ax.plot(*pos2, 'k+')
            ellipse2 = Ellipse(pos2,width=radius_x*2,height=radius_y*2,
                               angle=np.degrees(angle),facecolor='None',
                               edgecolor='k',linestyle='--')
            ax.add_patch(ellipse2)
        else:
            pos = sample_y.squeeze()
            ax.plot(*pos, 'k+',label='Function mean')
            ellipse = Ellipse(pos, width=radius_x*2, height=radius_y*2,
                                angle=np.degrees(angle),facecolor='None',
                                edgecolor='k',linestyle='--', 
                                label='True Noise contour')
            ax.add_patch(ellipse)
    if plot_continuous_preds:
        input = np.array([[x_bar]])
        # center the grid around the mean
        misc = Namespace(**vars(_TWO_D_MISC))
        y_bar = sample_y.squeeze()
        misc.x_range = np.array(misc.x_range) + y_bar[0]
        misc.y_range = np.array(misc.y_range) + y_bar[1]
        yy0, yy1, grid_size = prepare_grids(misc,mesh=True)
        X = data.input_to_torch_tensor(input, device)
        if hnet is not None:
            yy0_tensor = data.input_to_torch_tensor(yy0, device)
            yy1_tensor = data.input_to_torch_tensor(yy1, device)
            T = torch.stack((yy0_tensor,yy1_tensor),2)
            T = T.reshape((-1,2))

        if hnet is None:
            predictions = np.empty((data.out_shape[0],config.val_sample_size))
            predictions_var = np.ones((2,config.val_sample_size))\
                *(config.pred_dist_std**2)
            predictions_rho = np.zeros((1,config.val_sample_size))
            predictions_cov = np.vstack((predictions_var,predictions_rho))
        else:
            predictions = np.zeros((*grid_size[::-1], config.val_sample_size))
        
            
        for j in range(config.val_sample_size):              
            if hnet is None: # gl or ghl
                if config.mean_only:
                    Y = mnet.forward(X)
                else:
                    w_mean, w_rho = mnet.extract_mean_and_rho()
                    Y = mnet.forward(X, weights=None, mean_only=False,   
                                                        extracted_mean=w_mean, extracted_rho=w_rho,
                                                        disable_lrt=False)
                predictions[:,j] = Y[:,:2].detach().numpy()
                if log_likelihood:
                    predictions[:, j] = np.log(predictions[:, j])
                if Y.shape[1] == 5:
                    predictions_cov[:2,j] = np.exp(
                        Y[:,2:4].detach().cpu().numpy().squeeze())
                    predictions_cov[2,j] = np.tanh(
                        Y[:,4].detach().cpu().numpy().squeeze())
            else:
                if config.mean_only:
                    flow_weights = hnet.forward(X, ret_format='flattened')
                else:
                    w_mean, w_rho = hnet.extract_mean_and_rho(weights=None)
                    # Note, the sampling will happen inside the forward method.
                    flow_weights = hnet.forward(X, weights=None, mean_only=False,
                                                extracted_mean=w_mean,
                                                extracted_rho=w_rho,
                                                ret_format='flattened')
                flow_weights = flow_weights.repeat(np.prod(grid_size),1)
                Y, logabsdet = mnet.forward(T, weights=flow_weights)
                Y = mnet.log_p0(Y) + logabsdet
                if not log_likelihood:
                    Y = torch.exp(Y)
                Y = torch.reshape(Y, (*grid_size[::-1],))
                predictions[:, :, j] = Y.detach().cpu().numpy().squeeze()
        
        z = np.zeros((*grid_size[::-1],))
        yy = np.stack([yy0,yy1])
        for j in range(config.val_sample_size):
            if hnet is None: # gl or ghl
                z += mutils.normal_dist_2D(
                    yy,predictions[:,j],predictions_cov[:,j])
            else:
                z += predictions[...,j]
        z /= config.val_sample_size
        

        ###
                            
        
        args = [z]  
        kwargs = {'cmap': 'Blues', 'aspect': 'auto', 'origin': 'lower',
                'extent': (yy0[0,0], yy0[-1,-1], yy1[0,0], yy1[-1,-1]),
                'interpolation': 'bicubic'}
        f = ax.imshow(*args, **kwargs)
        if plot_marginal:
            cbar = plt.colorbar(f, ax=ax_y)
        else:
            cbar = plt.colorbar(f)
        if config.publication_style:
            cbar.ax.tick_params(labelsize=ts, length=lw, width=lw/2.)

        if plot_marginal:
            Y0 = yy0[0,:]
            Y1 = yy1[:,0]

            if plot_groundtruth:
                z_true_dist = np.zeros((*grid_size[::-1],))
                if isinstance(data,BivariateBimodalRegression):
                    z_true_dist += mutils.normal_dist_2D(yy,pos1,cov)
                    z_true_dist += mutils.normal_dist_2D(yy,pos2,cov)
                    z_true_dist /= z_true_dist.sum()
                else:
                    z_true_dist += mutils.normal_dist_2D(yy,pos,cov)
                # true marginal distribution, axis 0 corresponds to y, axis 1 corresponds to x
                z0_true = z_true_dist.sum(axis = 0)
                z1_true = z_true_dist.sum(axis = 1)
                Y0new, z0_true = mutils.interpolate(Y0,z0_true)
                Y1new, z1_true = mutils.interpolate(Y1,z1_true)
                ax_x.plot(Y0new,z0_true,'k--')
                ax_y.plot(z1_true,Y1new,'k--')

            # estimated marginal distribution
            z0 = z.sum(axis = 0)
            z1 = z.sum(axis = 1)
            Y0new, z0 = mutils.interpolate(Y0,z0)
            Y1new, z1 = mutils.interpolate(Y1,z1)

            ax_x.fill_between(Y0new,0,z0)
            ax_y.fill_betweenx(Y1new,0,z1)


    if publication_style:
        if plot_marginal:
            axes = (ax,ax_x,ax_y)
        else:
            axes = ax
        make_2D_for_publication(axes,plot_marginal=plot_marginal)
    else:
        plt.legend()
        plt.title(f'Conditional predictive distribution at $x={x_bar}$', 
                   fontsize=ts, pad=ts)
    ax.set_xlabel('$y_0$', fontsize=ts)
    ax.set_ylabel('$y_1$', fontsize=ts)

    figname = f'{figname}_x_eq_{x_bar}'

    if train_iter is not None:
        figname = figname + '_%s' % train_iter
    if figname:
        plt.savefig(figname + '.pdf', bbox_inches='tight')

    if testing_mode is not None:
        testing_mode = testing_mode + '/'
    sfx = ''
    writer.add_figure(testing_mode + 'test_predictions' + sfx, plt.gcf(),
                    train_iter, close=not config.show_plots)

    if config.show_plots:
        utils.repair_canvas_and_show_fig(plt.gcf())


def prepare_grids(misc, y_narrow=False, mesh = False):
    if isinstance(misc.grid_size, int):
        grid_size = (misc.grid_size, misc.grid_size)
    else:
        assert len(misc.grid_size) == 2
        grid_size = misc.grid_size
    y_range = misc.y_range_narrow if y_narrow else misc.y_range
    x = np.linspace(start=misc.x_range[0], stop=misc.x_range[1],
                    num=grid_size[0], dtype=np.dtype('f4'))
    y = np.linspace(start=y_range[0], stop=y_range[1],
                    num=grid_size[1], dtype=np.dtype('f4'))
    if not mesh:
        x_vis = np.linspace(start=misc.x_range_vis[0], stop=misc.x_range_vis[1],
                            num=grid_size[0], dtype=np.dtype('f4'))
        return x, x_vis, y, grid_size
    else:
        xx, yy = np.meshgrid(x,y)
        return xx, yy, grid_size


def kde_samples(x, x_range, bw=2):
    """Do kernel density estimation on a given sample.

    Args:
        x (np.array): The values of the samples.
        x_range (np.array): The range where to define the density.
        bw (float): The bandwidth to use for the kernel.

    Returns:
        (np.array): The probabilities in the defined range.
    """
    # Fit some model to the data.
    kde = KernelDensity(bandwidth=bw, kernel='gaussian')
    kde.fit(x[:, np.newaxis])

    # Sample probabilities for a range of outcomes
    p = kde.score_samples(x_range[:, np.newaxis])
    p = np.exp(p)

    return p

def get_ellipse_axis(cov):
    """
    Get contour ellipse of a bivariate distribution from its covariance matrix

    Args:
        cov (np.array): 2x2 covariance matrix.
    
    Returns:
        radius_x (float): half the width of the ellipse
        radius_y (float): half the height of the ellipse
        angle (float): angle in degrees by which the axis are rotated.
    """
    evals,E = np.linalg.eig(cov)
    radius_x,radius_y = np.sqrt(evals[0]),np.sqrt(evals[1])
    angle = np.degrees(np.arccos(E[0,0]))
    return radius_x,radius_y,angle


def get_y_samples(flow, weights, z_sample, grid_size, sample_size,
                  return_logabsdet=False):
    """Return y samples for given samples from the base distribution.

    This method returns a tensor of samples from p(y|w,x) for weights `w`
    generated by a list of `x` values. For each `x` (of which there are
    `grid_size` many), `sample_size` many samples of `y` are generated.

    Args:
        flow: Normalizing flow.
        weights: Weights generated by a hypernetwork for a vector of `x`
            values.
        z_sample: A tensor containing grid_size*sample_size many samples
            from the base distribution. If the same samples should be used
            for all `x` values, they must be aligned such that samples
            z_sample[i % grid_size] belong to the same `x` value.
        grid_size: The amount of `x` values used.
        sample_size: The amount of samples p(y|w,x) for a given `x`.

    Returns:
        y_samples: Tensor of size (grid_size, sample_size), containing
            samples from p(y|w,x).
    """
    # We need to use repeat() here because expand() only works for singleton
    # dimensions, i.e. if we have only a single sample that we want to repeat.
    # Also, repeat() copies the data whereas expand() only creates a new view,
    # but that doesn't matter here since we don't write to the tensor.
    repeated_weights = weights.repeat(sample_size, 1)
    y_samples, logabsdet = flow.inverse(z_sample, repeated_weights)
    y_samples = torch.reshape(y_samples, (sample_size, grid_size))
    y_reshaped = torch.transpose(y_samples, 0, 1)

    if return_logabsdet:
        logabsdet = torch.reshape(logabsdet, (sample_size, grid_size))
        logabsdet_reshaped = torch.transpose(logabsdet, 0, 1)
        return y_reshaped, logabsdet_reshaped

    return y_reshaped


def setup_conditional_fig(left=0.1, width = 0.65, bottom = 0.1, 
                          spacing = 0.005, figsize = (8,8)):
    """
    Prepares figure and axes for 2D conditional plots.
    
    Prepares figure and axes for plotting 2D conditional distributions in the 
    centre figure and their marginalized 1D distributions in the upper and 
    right figures.
    """
    # The rectangles for each plot.
    height = width
    marg = 1 - width - left - spacing
    rect_center = [left, bottom, width, height]
    rect_x = [left, bottom + height + spacing, width, marg]
    rect_y = [left + width + spacing, bottom, marg, height]

    # start with a square Figure
    fig = plt.figure(figsize=figsize)

    ax = fig.add_axes(rect_center)
    ax_x = fig.add_axes(rect_x, sharex=ax)
    ax_y = fig.add_axes(rect_y, sharey=ax)
    
    ax_x.tick_params(axis="x", labelbottom=False)
    ax_y.tick_params(axis="y", labelleft=False)
    
    x_visible_spines = ["bottom","left"]
    y_visible_spines = ["bottom","left"]

    for s in ax_x.spines:
        ax_x.spines[s].set_visible(s in x_visible_spines)
        ax_y.spines[s].set_visible(s in y_visible_spines)
    
    return fig,ax,ax_x,ax_y


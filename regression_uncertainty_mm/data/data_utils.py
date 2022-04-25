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
# @title          :oodregression/data/data_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :19/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for generating different datasets
--------------------------------------------------

A collection of helper functions for generating datasets to keep other scripts
clean.
"""
from argparse import Namespace
from hypnettorch.data.udacity_ch2 import UdacityCh2Data
from hypnettorch.utils import sim_utils as sutils
import hypnettorch.utils.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import random
from scipy import stats as spstats
from sklearn.mixture import GaussianMixture
import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms

from data.regression2d_gaussian import BivariateToyRegression
from data.regression2d_bimodal import BivariateBimodalRegression
from data.regression2d_heteroscedastic import BivariateHeteroscedasticRegression
from data.special.regression1d_data import ToyRegression
from data.special.regression1d_bimodal_data import BimodalToyRegression
import utils.math_utils as mutils

# Simple 1D regression grid.
_ONE_D_MISC = Namespace()
_ONE_D_MISC.x_range = [[-6.5], [6.5]]
_ONE_D_MISC.x_range_vis = [-6.5, 6.5]
_ONE_D_MISC.y_range = [-250, 250]
_ONE_D_MISC.y_range_narrow = [-80, 80]
_ONE_D_MISC.x = [1]
_ONE_D_MISC.grid_size = 200


# Simple 2D regression grid.
_TWO_D_MISC = Namespace()
_TWO_D_MISC.x_range = [-50, 50]
_TWO_D_MISC.x_range_vis = [-50, 50]
_TWO_D_MISC.y_range = [-50, 50]
_TWO_D_MISC.y_range_narrow = [-80, 80]
_TWO_D_MISC.x = [1]
_TWO_D_MISC.grid_size = 40

def generate_task(config, writer, dataset):
    """Generate a set of user defined tasks.

    Args:
        config: Command-line arguments.
        writer: Tensorboard writer, in case plots should be logged.
        dataset (str): The name of the dataset to be used.

    Returns:
        (....): Tuple containing:

        - **data_struct**: The data structure and data handler (for non
            steering angle experiments).
        - **tuple**: For the steering angle dataset, a tuple with train,
            validation and test loaders.
    """
    if dataset == 'toy_1D':
        train_inter = misc.str_to_ints(config.train_inter)
        test_inter = misc.str_to_ints(config.test_inter)
        val_inter = misc.str_to_ints(config.val_inter)
        num_val = config.val_set_size if config.val_set_size != 0 else None

        return (_generate_1d_task(show_plots=config.show_plots,
                                 data_random_seed=config.data_random_seed,
                                 writer=None if config.no_plots else writer,
                                 train_func=config.train_func,
                                 train_inter=train_inter,
                                 num_train=config.num_train, 
                                 test_inter=test_inter,
                                 num_test=config.num_test,
                                 val_inter=val_inter,
                                 num_val=num_val,
                                 std=np.sqrt(config.gaussian_variance),
                                 noise=config.noise), None)
    if dataset == 'toy_2D':
        train_inter = misc.str_to_ints(config.train_inter)
        test_inter = misc.str_to_ints(config.test_inter)
        val_inter = misc.str_to_ints(config.val_inter)
        num_val = config.val_set_size if config.val_set_size != 0 else None
        cov = parse_cov_matrix(config.cov_matrix)

        return (_generate_2d_task(show_plots=config.show_plots,
                                 data_random_seed=config.data_random_seed,
                                 writer=None if config.no_plots else writer,
                                 train_func=config.train_func,
                                 train_inter=train_inter,
                                 num_train=config.num_train, 
                                 test_inter=test_inter,
                                 num_test=config.num_test,
                                 val_inter=val_inter,
                                 num_val=num_val,
                                 noise=config.noise,
                                 cov=cov,
                                 offset=config.offset,
                                 rot_angle=config.rot_angle), None)
    if dataset == 'steering_angle':
        data, train_loader, val_loader, test_loader = open_udacity_ch2(config,
                                                      create_train_loader=True)
        return data, (train_loader, val_loader, test_loader)
    else:
        raise NotImplementedError


def _generate_1d_task(show_plots=True, data_random_seed=42, writer=None,
                      train_func='cube', train_inter=None, num_train=20,
                      test_inter=None, num_test=50, val_inter=None,
                      num_val=50, noise='gaussian', std=3):
    """Generate a set of tasks for 1D regression.

    Args:
        show_plots: Visualize the generated datasets.
        data_random_seed: Random seed that should be applied to the
            synthetic data generation.
        writer: Tensorboard writer, in case plots should be logged.
        train_func: Descriptor of 1D function to be learned.
        train_inter: Training interval.
        num_train: The number of training points.
        test_inter: Test interval.
        num_test: The number of test points.
        val_inter: validation interval.
        num_val: The number of validation points.
        std: The standard deviation of the Gaussian modes.
        noise (str, optional): The type of noise to be used.

    Returns:
        data_handler: A data handler.
    """
    if train_inter is None:
        train_inter = [-4, 4]
    if test_inter is None:
        test_inter = [-6, 6]
    if num_val is not None:
        if val_inter is None:
            val_inter = train_inter
    else:
        val_inter = None

    funcs = {'cube': lambda x: (x ** 3.),
             'wavelet': lambda x: (80 * np.sin(x) * np.exp(-np.abs(x))),
             'sin': lambda x: (80 * np.sin(1.5*x))}

    if train_func not in funcs:
        raise NotImplementedError('Regression function %s not implemented!'
                                  % train_func)

    if noise == 'bimodal':
        dhandler = BimodalToyRegression(train_inter=train_inter,
            num_train=num_train, val_inter=val_inter, num_val=num_val,
            test_inter=test_inter, num_test=num_test,
            map_function=funcs[train_func], dist1=50, std1=std,
            rseed=data_random_seed, perturb_test_val=True)
    elif noise == 'gaussian':
        dhandler = ToyRegression(train_inter=train_inter, num_train=num_train,
                             test_inter=test_inter, num_test=num_test,
                             val_inter=val_inter, num_val=num_val,
                             map_function=funcs[train_func], std=std,
                             rseed=data_random_seed, perturb_test_val=True)
    else:
        raise NotImplementedError('Noise type "%s" not implemented.' % noise)

    if writer is not None:
        dhandler.plot_dataset(show=False)
        writer.add_figure('task/dataset', plt.gcf(), close=not show_plots)
        if show_plots:
            misc.repair_canvas_and_show_fig(plt.gcf())
    elif show_plots:
        dhandler.plot_dataset()

    return dhandler

def _generate_2d_task(show_plots=True, data_random_seed=42, writer=None,
                      train_func='cube', train_inter=None, num_train=20,
                      test_inter=None, num_test=50, val_inter=None, num_val=50,
                      noise='gaussian', cov=9, offset=None, rot_angle=None):
    """Generate a set of tasks for 1D regression.

    Args:
        show_plots: Visualize the generated datasets.
        data_random_seed: Random seed that should be applied to the
            synthetic data generation.
        writer: Tensorboard writer, in case plots should be logged.
        train_func: Descriptor of 1D function to be learned.
        train_inter: Training interval.
        num_train: The number of training points.
        test_inter: Test interval.
        num_test: The number of test points.
        val_inter: validation interval.
        num_val: The number of validation points.
        cov_matrix: cov_matrix of the noise.
        noise (str, optional): The type of noise to be used.
        offset: half the distance between the two modes of a bimodal 
            noise.
        rot_angle: Rotation angle for the bimodal noise.

    Returns:
        data_handler: A data handler.
    """
    if train_inter is None:
        train_inter = [-4, 4]
    if test_inter is None:
        test_inter = [-6, 6]
    if val_inter is None:
        val_inter = train_inter
    funcs = {'cube': lambda x: (x ** 3.),
             'wavelet': lambda x: (80 * np.sin(x) * np.exp(-np.abs(x))),
             'sin': lambda x: (80 * np.sin(1.5*x))}

    if train_func not in funcs:
        raise NotImplementedError('Regression function %s not implemented!'
                                  % train_func)
    
    map_function = lambda x:np.hstack(
        [funcs[train_func](x),funcs[train_func](x)])

    if noise == 'gaussian':
        dhandler = BivariateToyRegression(train_inter=train_inter, 
                             num_train=num_train, test_inter=test_inter, 
                             num_test=num_test, val_inter=val_inter, 
                             num_val=num_val, map_function=map_function, 
                             cov=cov, rseed=data_random_seed)
    elif noise == 'bimodal':
        dhandler = BivariateBimodalRegression(train_inter=train_inter, 
                             num_train=num_train, test_inter=test_inter, 
                             num_test=num_test, val_inter=val_inter, 
                             num_val=num_val, map_function=map_function, 
                             cov=cov, offset=offset, rot_angle=rot_angle, 
                             rseed=data_random_seed)
    elif noise == 'heteroscedastic':
        dhandler = BivariateHeteroscedasticRegression(train_inter=train_inter, 
                             num_train=num_train, test_inter=test_inter, 
                             num_test=num_test, val_inter=val_inter, 
                             num_val=num_val, map_function=map_function, 
                             cov=cov, rseed=data_random_seed)
    
    return dhandler

def parse_cov_matrix(cov_str):
    """
    Parse a command line string cov matrix into a list of floats. 
    """
    if not isinstance(cov_str,str):
        "Assuming it is then already in valid form"
        return cov_str
    else:
        tokens = cov_str.split(',')
        tokens_float = [float(a.replace('[','').replace(']','')) for a in tokens]
        if len(tokens_float) == 1:
            return tokens_float[0]
        if len(tokens_float) == 2:
            return [tokens_float[0],tokens_float[1]]
        if len(tokens_float) == 4:
            return [
                [tokens_float[0],tokens_float[1]],
                [tokens_float[2],tokens_float[3]]
            ]
        raise ValueError('cov_matrix should contain 1,2 or 4 floats, '+ \
            f'but {len(tokens_float)} were given.')
def open_udacity_ch2(config, create_train_loader=True):
    """Open the UdacityCh2 dataset and create the corresponding dataloaders.

    Args:
        config: Command-line options.
        batch_size: Batchsize for all data loaders.
        create_train_loader (bool): Whether a loader for the training set should
            be generated.

    Returns:
        (tuple): Tuple containing:

        ``[data_handler, train_loader, val_loader, test_loader]``
        Note, ``train_loader`` and ``val_loader`` might be ``None``.
    """
    data = UdacityCh2Data(config.data_dir,
                          num_val=config.val_set_size)

    if create_train_loader:
        train_loader = DataLoader(data.torch_train,
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True,
            worker_init_fn=_dloader_worker_init(config, change_rseed=True))
    else:
        train_loader = None

    # Validation loader is treated like a test loader (see data augmentation
    # of UdacityCh2Data class).
    if data.num_val_samples > 0:
        val_loader = DataLoader(data.torch_val,
            batch_size=config.val_batch_size, shuffle=False,
            num_workers=config.num_workers, pin_memory=True,
            worker_init_fn=_dloader_worker_init(config))
    else:
        val_loader = None

    test_loader = DataLoader(data.torch_test, batch_size=config.val_batch_size,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
        worker_init_fn=_dloader_worker_init(config))

    return data, train_loader, val_loader, test_loader

def _dloader_worker_init(config, change_rseed=False):
    """Provides an initialization function for the workers of an PyTorch
    dataloader.

    The returned function ensures:
        - Every worker will have a different random seed. Note, this will
          not make the overall computation deterministic, as the workers
          are still scheduled by the operating system. Though, it can ensure
          that computations are similarly pseudo-random as when using no
          multi processing. For instance, assume data augmentation is
          applied to images. If the seed was just copied to all workers,
          then the same random augmentation will be applied several times.

    Args:
        config: Command-line arguments.
        change_rseed (bool): Whether the random seed of each worker
            should be explicitly set to a different value.
            In the normal case, randomness (stemming from random seeds, not
            from hardware) should only affect a training set loader.

    Returns:
        A function handle, that can be passed to argument ``worker_init_fn``
        of class :class:`torch.utils.data.DataLoader`.
    """
    def _w_init(w_id):
        if change_rseed:
            wseed = config.random_seed + w_id

            torch.manual_seed(wseed)
            torch.cuda.manual_seed_all(wseed)
            np.random.seed(wseed)
            random.seed(wseed)

    return _w_init

def test_splitting_image(mnet, data, device, config, shared, logger, writer,
                         mode, hnet=None, epoch=None):
    """Test the networks on an image with splitting lanes.

    Args:
        mnet: The main network.
        device: The device.
        config: The configuration.
        shared: Namespace with useful information.
        logger: The logger.
        writer: The tensorboard writer.
        mode: The mode of the experiment (`gl`, `ghl` or `nf`).
        hnet: The (optional) hypernetwork.
    """
    # If the hypernetwork is provided, we are in an NF experiment, and the means
    # and variances refer to it.
    logger.info('Testing on split lane image...')
    with torch.no_grad():

        image_path1='split_lane/rafael_swiss_road_256x192.png'
        image_path2='split_lane/rafael_swiss_road_256x192_2.png'

        # Load the image.
        X1 = load_image(image_path1).to(device)
        X2 = load_image(image_path2).to(device)

        # Plot the predictions.
        figname = None
        if not config.no_plots:
            fig_dir = os.path.join(config.out_dir, 'figures')
            figname = os.path.join(fig_dir, 'split_lane_prediction')
        if epoch is not None:
            figname += '_epoch%i' % epoch
        from data import plotting_utils as plt_utils
        plt_utils.plot_pred_dist_steering(config, writer, data, mnet, device,
                                hnet=hnet, figname=figname,
                                publication_style=config.publication_style,
                                plot_groundtruth=False, plot_images=False,
                                input_images=[X1, X2], save_predictions=True)

def load_image(image_path, folder_path='../data/splitting_lane_images/'):
    """Load a certain image and make it suitable for torch processing.

    Args:
        folder_path (str): The path to the container folder. Note that this
            folder needs to have subfolders for images to be properly loaded.
        image_path (str): The path to the image within the container folder.
    """
    img_name = os.path.join(folder_path, image_path)
    _, test_transform = UdacityCh2Data.torch_input_transforms()
    img_folder = datasets.ImageFolder(folder_path, test_transform)
    X = img_folder.loader(img_name)
    trans = transforms.ToTensor()
    X = trans(X).unsqueeze(dim=0) # add a batch dimension
    # X = X.permute(0, 1, 3, 2) # make compatible to how images are normally

    return X

def compute_1D_uncertainty_stats(config, writer, data, mnet, device,
                                 hnet=None, log_likelihood=False):
    """Compute uncertainty statistics for 1D toy regression experiments.

    In particular, this function computes the mean, the variance, the entropy,
    the multi-modality score and the model disagreement (for Bayesian models)
    at a specific input location determined by the input ``x``.

    Args:
        config: Command-line arguments.
        writer: Tensorboard summary writer.
        data: A data loader.
        mnet: The network.
        device: The cuda device.
        hnet: The (optional) hypernetwork. Provided when the model is a NF.
        log_likelihood (bool): Whether to plot the likelihood in a log scale.

    Returns:
        (dict): Dictionary with all relevant statistics at the requested point.
    """
    stats = {'model': {'mean': -1, 'variance': -1, 'entropy': -1,
                 'multimodality': {'p':-1,'d':-1,'isunimodal':-1},
                 'model_disagreement': -1, 'det': config.mean_only},
             'ground_truth': {'mean': -1, 'variance': -1, 'entropy': -1,
                 'multimodality': {'p':-1,'d':-1,'isunimodal':-1}, 
                 'num_train': config.num_train}}

    ### Prepare the data.
    # This function is only valid at the origin, since we know there what the
    # ground-truth means and variances are. Else we'd have to go to the data
    # generating functions and compute it.
    x = np.array([0])
    y = np.linspace(*_ONE_D_MISC.y_range, 1000)
    x_torch = data.input_to_torch_tensor(x, device)
    X = data.input_to_torch_tensor(x, device)
    if hnet is not None:
        # Need a matrix with all y values at which to evaluate the density.
        T = data.input_to_torch_tensor(y, device)
        T = torch.vstack([T] * 1)
        T = torch.transpose(T, 0, 1)
        T = torch.reshape(T, (-1,))

    ### Make predictions.
    if hnet is None:
        predictions = np.zeros(config.val_sample_size)
        predictions_std = np.ones(config.val_sample_size) \
            * config.pred_dist_std # default std, overwritten below for ghl
    else:
        predictions = np.zeros((len(y), config.val_sample_size))
 
    for j in range(config.val_sample_size):
        if hnet is None: # gl or ghl
            if config.mean_only:
                Y = mnet.forward(X)
            else:
                w_mean, w_rho = mnet.extract_mean_and_rho()
                Y = mnet.forward(X, weights=None, mean_only=False,
                                 extracted_mean=w_mean, extracted_rho=w_rho,
                                 disable_lrt=False)
            predictions[j] = Y[0].detach().cpu().numpy().squeeze()
            if log_likelihood:
                predictions[j] = np.log(predictions[j])

            if len(Y) == 2:
                rho = Y[1].detach().cpu().numpy().squeeze()
                predictions_std[j] = np.exp(0.5 * rho)   
            
            # Store parameters to compute model disagreement.
            if j == 0:
                params = Y.clone().detach()
            else:
                params = torch.vstack((params, Y.detach().squeeze()))
        else:
            X = X.unsqueeze(dim=0)
            if config.mean_only:
                flow_weights = hnet.forward(X, ret_format='flattened')
            else:
                w_mean, w_rho = hnet.extract_mean_and_rho(weights=None)
                # Note, the sampling will happen inside the forward method.
                flow_weights = hnet.forward(X, weights=None, mean_only=False,
                                            extracted_mean=w_mean,
                                            extracted_rho=w_rho,
                                            ret_format='flattened')
            flow_weights = flow_weights.repeat(len(y), 1)
            Y, logabsdet = mnet.forward(T, weights=flow_weights)
            Y = mnet.log_p0(Y) + logabsdet
            if not log_likelihood:
                Y = torch.exp(Y)
            Y = torch.reshape(Y, (len(y), 1))
            predictions[:, j] = Y.detach().cpu().numpy().squeeze()

            # Store parameters to compute model disagreement.
            if j == 0:
                params = flow_weights.clone().detach()
            else:
                params = torch.vstack((params, flow_weights.detach().squeeze()))

    ### Determine the density, averaged across models.
    z_indv = np.zeros((len(y), config.val_sample_size))
    for j in range(config.val_sample_size):
        if hnet is None: # gl or ghl
            z_indv[:, j] = mutils.normal_dist(y, predictions[j],
                                              predictions_std[j])
        else:
            z_indv[:, j] = predictions[:, j]
    z = z_indv.mean(axis=1) # average across models
    # Convert to an actual density so that it sums up to one.
    z /= (z.mean()*len(y))

    ### Obtain statistics.
    if hnet is None and config.mean_only:
        # For Gaussian likelihood based deterministic models we can compute
        # things analytically.
        stats['model']['mean'] = predictions 
        stats['model']['variance'] = predictions_std**2
        stats['model']['entropy'] = \
                    0.5 * np.log(2 * np.pi * predictions_std**2) + 0.5
    else:
        model_mean = (z * y).sum()
        stats['model']['mean'] = model_mean
        stats['model']['variance'] = (z * (y - model_mean)**2).sum()
        stats['model']['entropy'] = - (z * np.log(z)).sum()
        if config.val_sample_size > 1:
            stats['model']['model_disagreement'] = \
                                        compute_model_disagreement(params)
    ## Ground-truth for sanity checks.
    # ground_truth_y = mutils.normal_dist(y, -50, 3)*0.5 + \
    #                  mutils.normal_dist(y, 50, 3)*0.5
    p, d, is_unimodal = compute_multimodality_score(z)
    stats['model']['multimodality']['p'] = p #TODO. take the smallest one?
    stats['model']['multimodality']['d'] = d
    stats['model']['multimodality']['isunimodal'] = is_unimodal

    ### Obtain statistics for the ground-truth.
    stats['ground_truth']['mean'] = 0
    if config.noise == 'gaussian':
        stats['ground_truth']['variance'] = config.gaussian_variance
        stats['ground_truth']['entropy'] = \
                    0.5 * np.log(2 * np.pi * config.gaussian_variance) + 0.5
        stats['ground_truth']['multimodality']['p'] = 1 #TODO should we take 1 or 0
        stats['ground_truth']['multimodality']['d'] = 0
        stats['ground_truth']['multimodality']['isunimodal'] = True
    elif config.noise == 'bimodal':
        # The variance can be computed analytically according to:
        # https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        stats['ground_truth']['variance'] = \
            0.5* (config.gaussian_variance + config.gaussian_variance) + \
            0.25 * (50 - (-50))**2
        # WARNING! This is hardcoded as it need to be estimated from data.
        stats['ground_truth']['entropy'] = 3.20409
        stats['ground_truth']['multimodality']['p'] = data._alpha1
        stats['ground_truth']['multimodality']['d'] = \
            (data._dist1 + data._dist2)/(2*np.sqrt(data._std1*data._std2))
        stats['ground_truth']['multimodality']['isunimodal'] = False

    ### Store.
    with open(os.path.join(config.out_dir, 'uncertainty_stats.pickle'), 'wb') \
                    as handle:
        pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compute_multimodality_score(z):
    r"""Compute the uni-modality score of the predictive distribution.

    Fits a mixture of two Gaussians to the input samples and 
    tests the resulting mixture for unimodality and 
    returns scores of unimodality.

    Args:
        z (np.array): The predictions at the origin for different y values,
            averaged across models.

    Returns:
        p (float): The mixing factor between the two Gaussians.
        d (float): A score of the overlap between the two Gaussians.
            The formula is given by: 
            ..math::
                d = \frac{|\mu_1-\mu_2|}{2\sqrt{\sigma_1 \sigma_2}}
        is_unimodal (bool): Whether the mixture is unimodal.
    """
    gaussian_mixture = GaussianMixture(n_components=2).fit(z.reshape(-1,1))
    mu1 = gaussian_mixture.means_[0,0]
    mu2 = gaussian_mixture.means_[1,0]
    sigma1 = np.sqrt(gaussian_mixture.covariances_[0,0,0])
    sigma2 = np.sqrt(gaussian_mixture.covariances_[1,0,0])
    p = gaussian_mixture.weights_[0]
    d = np.abs(mu1-mu2)/(2*np.sqrt(sigma1*sigma2))
    is_unimodal = d <= 1
    is_unimodal |= np.abs(np.log(1-p) - np.log(p)) >= 2 * \
        np.log(d-np.sqrt(d**2-1)) + 2*d*np.sqrt(d**2 - 1)
    return p, d, is_unimodal

def compute_model_disagreement(params):
    r"""Compute the model disagreement.

    The model disagreement is computed here as standard deviation of the
    parameters of the likelihood averaged over the dimensionality of the space
    to obtain a scalar. In the manuscript it is called $\mathcal{U}_V(x)$.

    Args:
        params(torch.tensor): This are the parameters of the likelihood model. 
            - Mean for GLc
            - Mean and variance concatenated for GL 
            - Weights for the Normalizing flow model 

    Returns:
        (float): The model disagreement.
    """
    md = torch.std(params, dim=0).mean().cpu().numpy()
    return md

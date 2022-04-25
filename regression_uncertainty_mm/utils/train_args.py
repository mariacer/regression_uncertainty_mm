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
# @title          :utils/train_args.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :20/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
CLI Argument Parsing for all experiments
-----------------------------------------

Command-line arguments common to all experiments.
"""
import warnings

def miscellaneous_args(agroup, show_mean_only=True, show_use_logvar_enc=True,
                       show_disable_lrt_test=True):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the miscellaneous argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.miscellaneous_args`.
        show_mean_only (bool): Whether the option `show_mean_only` should be
            provided.
        show_use_logvar_enc (bool): Whether the option
            `use_logvar_enc` should be provided.
        show_disable_lrt_test (bool): Whether the option
            `disable_lrt_test` should be provided.
    """
    if show_mean_only:
        agroup.add_argument('--mean_only', action='store_true',
                            help='Train deterministic network. Note, option ' +
                                 '"kl_scale" needs to be zero in this case, ' +
                                 'as no prior-matching can be applied.')
    if show_use_logvar_enc:
        agroup.add_argument('--use_logvar_enc', action='store_true',
                            help='Use the log-variance encoding for the ' +
                                 'variance parameters of the Gaussian weight ' +
                                 'posterior.')
    if show_disable_lrt_test:
        agroup.add_argument('--disable_lrt_test', action='store_true',
                            help='If activated, the local-reparametrization ' +
                                 'trick will be disabled during testing, ' +
                                 'i.e., all test samples are processed using ' +
                                 'the same set of models.')
    agroup.add_argument('--store_models', action='store_true',
                        help='Whether the models should be checkpointed at ' +
                             'end of training (or at the end of each epoch, ' +
                             'if there is more than one.')
    agroup.add_argument('--no_plots', action='store_true',
                            help='If activated, no plots will be generated ' +
                                 '(not even for the writer). This can be ' +
                                 'useful to save time, e.g. for a ' +
                                 'hyperparameter search.')

          
def mc_args(tgroup, vgroup, show_train_sample_size=True, dval_sample_size=100):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the training and validation argument group for options
    specific to the Monte-Carlos sampling procedure used to approximate the loss
    and the predictive distribution.

    Args:
        tgroup: The argument group returned by method
            :func:`utils.cli_args.train_args`.
        vgroup: The argument group returned by method
            :func:`utils.cli_args.eval_args`.
        show_train_sample_size (bool): Whether option `show_train_sample_size`
            should be shown.
        dval_sample_size (int): Default value of option `val_sample_size`.
    """
    if show_train_sample_size:
        tgroup.add_argument('--train_sample_size', type=int, metavar='N',
                            default=100,
                            help='How many samples should be used for the ' +
                                 'approximation of the negative log ' +
                                 'likelihood in the loss. ' +
                                 'Default: %(default)s.')
    vgroup.add_argument('--val_sample_size', type=int, metavar='N',
                        default=dval_sample_size,
                        help='How many weight samples should be drawn to ' +
                             'calculate an MC sample of the predictive ' +
                             'distribution. Default: %(default)s.')


def train_args(tgroup, show_local_reparam_trick=True, show_pred_dist_std=True,
               show_kl_scale=False):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the training argument group specific to training
    probabilistic models.

    Args:
        tgroup: The argument group returned by function
            :func:`utils.cli_args.train_args`.
        show_local_reparam_trick (bool): Whether the option
            `local_reparam_trick` should be provided.
        show_kl_scale (bool): Whether the option
            `kl_scale` should be provided.
    """
    tgroup.add_argument('--prior_variance', type=float, default=1.0,
                        help='Variance of the Gaussian prior. ' +
                             'Default: %(default)s.')
    if show_local_reparam_trick:
        tgroup.add_argument('--local_reparam_trick',action='store_true',
                            help='Use the local reparametrization trick.')
    if show_pred_dist_std:
        tgroup.add_argument('--pred_dist_std', type=float, default=3,
                            help='The standard deviation of the likelihood ' +
                                 'distribution. Note, this value should be ' +
                                 'fixed but reasonable for a given dataset.' +
                                 'Default: %(default)s.')
    if show_kl_scale:
        tgroup.add_argument('--kl_scale', type=float, default=1.,
                        help='A scaling factor for the prior matching term ' +
                             'in the variational inference loss. NOTE, this ' +
                             'option should be used with caution as it is ' +
                             'not part of the ELBO when deriving it ' +
                             'mathematically. ' +
                             'Default: %(default)s.')


def simple_regression_args(parser, dgaussian_variance=9.):
    """Adds simple 1D regression specific arguments to an argument group.

    Args:
        agroup (ArgumentGroup): An argument group to which the additional
            arguments are added.
        dgaussian_variance (float): Default value of option `gaussian_variance`.
    """
    agroup = parser.add_argument_group('1D regression options')
    agroup.add_argument('--train_func', type=str, default='cube',
                        help='1D regression function. Default: %(default)s.',
                        choices=['cube', 'wavelet', 'sin'])
    agroup.add_argument('--train_inter', type=str, default='-4,4',
                        help='Training interval. Default: %(default)s.')
    agroup.add_argument('--num_train', type=int, default=20, metavar='N',
                        help='Number of training points. Default: %(default)s.')
    agroup.add_argument('--test_inter', type=str, default='-6,6',
                        help='Test interval. Default: %(default)s.')
    agroup.add_argument('--num_test', type=int, default=50, metavar='N',
                        help='Number of test points. Default: %(default)s.')
    agroup.add_argument('--val_inter', type=str, default='-4,4',
                        help='Validation interval. Default: %(default)s.')
    agroup.add_argument('--gaussian_variance', type=float, 
                        default=dgaussian_variance,
                        help='The variance of the Gaussian to be used. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--noise', type=str, default='gaussian',
                        help='Type of noise which is added to the data. ' +
                             'Default: %(default)s.',
                        choices=['gaussian', 'bimodal'])
    agroup.add_argument('--store_uncertainty_stats', action='store_true',
                        help='Whether uncertainty statistics at the origin ' +
                             '(`x=0`) should be pickled. This is useful to ' +
                             'later on make plots.')


def twoD_regression_args(parser):
    """Adds simple 2D regression specific arguments to an argument group.

    Args:
        agroup (ArgumentGroup): An argument group to which the additional
            arguments are added.
    """
    agroup = parser.add_argument_group('2D regression options')
    agroup.add_argument('--train_inter', type=str, default='-4,4',
                        help='Training interval. Default: %(default)s.')
    agroup.add_argument('--num_train', type=int, default=20, metavar='N',
                        help='Number of training points. Default: %(default)s.')
    agroup.add_argument('--test_inter', type=str, default='-6,6',
                        help='Test interval. Default: %(default)s.')
    agroup.add_argument('--num_test', type=int, default=50, metavar='N',
                        help='Number of test points. Default: %(default)s.')
    agroup.add_argument('--val_inter', type=str, default='-4,4',
                        help='Validation interval. Default: %(default)s.')
    agroup.add_argument('--train_func', type=str, default='cube',
                        help='2D regression function (same in both ' +
                             'directions). Default: %(default)s.',
                        choices=['cube', 'wavelet', 'sin'])
    agroup.add_argument('--noise', type=str, default='gaussian',
                        help='Noise added to data. Default: %(default)s.',
                        choices=['gaussian', 'bimodal','heteroscedastic'])
    agroup.add_argument('--cov_matrix', type=str, default='[10,2]',
                        help='Covariance of noise added. If noise is bimodal, '+ 
                             'same covariance for both modes  Default: ' +
                             '%(default)s.')
    agroup.add_argument('--offset', type=float, default=3,
                        help='half the distance between the mean of the ' +
                             'two modes. Default: %(default)s.')
    agroup.add_argument('--rot_angle', type=float, default=0.78,
                        help='Number of validation points. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--permute_flow_layers', type=str, default=None,
                        help='list of permutation flags for 2d nf layers. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--conditioner_arch', type=str, default='10',
                        help='list of layer sizes of the 2d nf conditioner ' +
                             'hnet. Default: %(default)s.')
    

def steering_angle_prediction_args(parser):
    """Adds steering angle prediction specific arguments to an argument group.

    Args:
        agroup (ArgumentGroup): An argument group to which the additional
            arguments are added.
    """
    agroup = parser.add_argument_group('Steering angle prediction options')
    agroup.add_argument('--use_empty_test_set', action='store_true',
                        help='Even if the test set is empty, this option ' +
                             'causes the loss to be evaluated on it.')
    agroup.add_argument('--num_plotted_predictions', type=int, default=4,
                        help='Number of inputs for which to plot predictions.')
    agroup.add_argument('--plot_steering_images', action='store_true',
                        help='If ``True`` actual input images for the ' +
                             'steering angle prediction task will be ' +
                             'plotted. Else, only predictions are plotted.')


def check_invalid_args_general(config):
    """Sanity check for BbB command-line arguments.

    Args:
        config: Parsed command-line arguments.
    """
    # Not mathematically correct, but might be required if prior is not
    # appropriate.
    if hasattr(config, 'kl_scale') and  config.kl_scale != 1.0:
        warnings.warn('Prior matching term will be scaled by %f.'
                      % config.kl_scale)
    if hasattr(config, 'local_reparam_trick'):
        if config.local_reparam_trick:
            if hasattr(config, 'dropout_rate') and config.dropout_rate != -1:
                raise ValueError('Dropout not implemented for network with ' +
                                 'local reparametrization trick.')
            if hasattr(config, 'specnorm') and config.specnorm:
                raise ValueError('Spectral norm not implemented for network ' +
                                 'with local reparametrization trick.')
            if hasattr(config, 'batchnorm') and config.batchnorm or \
                    hasattr(config, 'no_batchnorm') and not config.no_batchnorm:
                raise ValueError('Batchnorm not implemented for network ' +
                                 'with local reparametrization trick.')
        else:
            if config.disable_lrt_test:
                warnings.warn('Option "disable_lrt_test" has no effect if ' +
                              'the local-reparametrization trick is not used.')

    if hasattr(config, 'mean_only') and config.mean_only:
        if hasattr(config, 'kl_scale') and config.kl_scale != 1:
            warnings.warn('Prior-matching is not applicable for ' +
                          'deterministic networks. Setting "kl_scale" to zero.')
            config.kl_scale = 0
        if hasattr(config,'local_reparam_trick') and config.local_reparam_trick:
            raise ValueError('Local-reparametrization trick cannot be ' +
                             'applied to non-Gaussian networks.')
        if config.train_sample_size > 1:
            warnings.warn('A "train_sample_size" greater than 1 doesn\'t ' +
                          'make sense for a deterministic network. ' +
                          'Overwritting to `1`.')
            config.train_sample_size = 1
        if config.val_sample_size > 1:
            warnings.warn('A "val_sample_size" greater than 1 doesn\'t ' +
                          'make sense for a deterministic network. ' +
                          'Overwritting to `1`.')
            config.val_sample_size = 1
        if config.disable_lrt_test:
            warnings.warn('Option "disable_lrt_test" not applicable to ' +
                          'deterministic networks.')
        if config.use_logvar_enc:
            warnings.warn('Option "use_logvar_enc" not applicable to ' +
                          'deterministic networks.')

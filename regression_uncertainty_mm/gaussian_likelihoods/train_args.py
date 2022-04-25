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
# @title          :gaussian_likelihoods/train_args.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :20/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
CLI Argument Parsing for Gaussian likelihood experiments
--------------------------------------------------------

Command-line arguments and default values for the Gaussian likelihood
experiments.
"""
import argparse
from datetime import datetime
import hypnettorch.utils.cli_args as cli
import warnings

import utils.train_args as common_args

def parse_cmd_arguments(mode='gl', dataset='toy_1D', default=False, argv=None):
    """Parse command-line arguments.

    Args:
        mode (str): The CLI mode of the experiment.
        dataset (str, optional): The dataset to be used.
        default (optional): If ``True``, command-line arguments will be ignored
            and only the default values will be parsed.
        argv (optional): If provided, it will be treated as a list of command-
            line argument that is passed to the parser in place of sys.argv.

    Returns:
        The Namespace object containing argument names and values.
    """
    curr_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if dataset == 'toy_1D':
        description = 'Simple 1D Regression - '
        dout_dir = './out_1D/'
    elif dataset == 'toy_2D':
        description = 'Simple 2D Regression - '
        dout_dir = './out_2D/'
    elif dataset == 'steering_angle':
        description = 'Steering angle prediction - '
        dout_dir = './out_steering_angle/'
    else:
        raise ValueError('Dataset "%s" unknown.' % (dataset))

    if mode == 'gl':
        description += 'Gaussian Likelihood'
        dout_dir += 'gl/run_' + curr_date
    elif mode == 'ghl':
        description += 'Gaussian Heteroscedastic Likelihood'
        dout_dir += 'ghl/run_' + curr_date
    else:
        raise ValueError('Mode "%s" unknown.' % (mode))

    parser = argparse.ArgumentParser(description=description)

    #################
    ### Arguments ###
    #################

    # Dataset arguments.
    if dataset == 'toy_1D':
        ######################
        ### 1D Toy Dataset ###
        ######################
        common_args.simple_regression_args(parser)
        # Miscellaneous arguments.
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=True,
            show_publication_style=True, dout_dir=dout_dir)
        common_args.miscellaneous_args(misc_agroup)
        # Evaluation arguments.
        eval_agroup = cli.eval_args(parser, dval_iter=250, dval_set_size=50,
                                    show_val_set_size=True)
        # Training arguments.
        train_agroup = cli.train_args(parser, show_lr=True, dn_iter=10001,
                                      dlr=1e-2, show_adam_beta1=True,
                                      show_momentum=True,
                                      show_clip_grad_value=True,
                                      show_clip_grad_norm=True)
        common_args.train_args(train_agroup, show_kl_scale=True)
        # Main network arguments.
        mnet_group = cli.main_net_args(parser, allowed_nets=['mlp'],
                                       dmlp_arch='10,10', dnet_act='sigmoid',
                                       show_no_bias=True)
        # Initialization arguments.
        init_agroup = cli.init_args(parser, custom_option=False)
        init_args(init_agroup)
        common_args.mc_args(train_agroup, eval_agroup)
        main_net_args(mnet_group, allowed_nets=['mlp'])

    elif dataset == 'toy_2D':
        ######################
        ### 2D Toy Dataset ###
        ######################
        common_args.twoD_regression_args(parser)
        # Miscellaneous arguments.
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
            synthetic_data=True, show_plots=True, no_cuda=True,
            show_publication_style=True, dout_dir=dout_dir)
        common_args.miscellaneous_args(misc_agroup)
        # Evaluation arguments.
        eval_agroup = cli.eval_args(parser, dval_iter=250, dval_set_size=50,
                                    show_val_set_size=True)
        # Training arguments.
        train_agroup = cli.train_args(parser, show_lr=True, dn_iter=10001,
                                      dlr=1e-2, show_adam_beta1=True,
                                      show_momentum=True,
                                      show_clip_grad_value=True,
                                      show_clip_grad_norm=True)
        common_args.train_args(train_agroup, show_kl_scale=True)
        # Main network arguments.
        mnet_group = cli.main_net_args(parser, allowed_nets=['mlp'],
                                       dmlp_arch='10,10', dnet_act='sigmoid',
                                       show_no_bias=True)
        # Initialization arguments.
        init_agroup = cli.init_args(parser, custom_option=False)
        init_args(init_agroup)
        common_args.mc_args(train_agroup, eval_agroup)
        main_net_args(mnet_group, allowed_nets=['mlp'])

    elif dataset == 'steering_angle':
        ########################
        ### Steering Dataset ###
        ########################
        common_args.steering_angle_prediction_args(parser)
        # Miscellaneous arguments.
        misc_agroup = cli.miscellaneous_args(parser, big_data=True,
            synthetic_data=False, show_plots=True, no_cuda=False,
            show_publication_style=True, dout_dir=dout_dir)
        common_args.miscellaneous_args(misc_agroup)
        # Evaluation arguments.
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                      dval_batch_size=128, dval_iter=1000,
                      show_val_set_size=True, dval_set_size=5000)
        # Training arguments.
        train_agroup = cli.train_args(parser, show_lr=True, dn_iter=1000,
                                      dlr=1e-3, dbatch_size=32,
                                      show_use_adam=True,
                                      show_use_rmsprop=True,
                                      show_use_adadelta=True,
                                      show_use_adagrad=True,
                                      show_epochs=True, depochs=100,
                                      show_clip_grad_value=True,
                                      show_clip_grad_norm=True)
        common_args.train_args(train_agroup, show_kl_scale=True)
        # Data arguments.
        cli.data_args(parser, show_data_dir=True,
                      ddata_dir='../datasets/udacity_ch2')
        # Main network arguments.
        mnet_group = cli.main_net_args(parser,
                    allowed_nets=['iresnet', 'lenet', 'resnet', 'wrn', 'mlp'],
                    show_no_bias=True, show_batchnorm=True,
                    show_no_batchnorm=False, show_bn_no_running_stats=True,
                    show_bn_distill_stats=False,
                    show_bn_no_stats_checkpointing=True,
                    show_specnorm=False, show_dropout_rate=True,
                    ddropout_rate=-1, show_net_act=True)
        # Initialization arguments.
        init_agroup = cli.init_args(parser, custom_option=False)
        init_args(init_agroup)
        common_args.mc_args(train_agroup, eval_agroup)

    ##################################
    ### Finish-up Argument Parsing ###
    ##################################
    # Including all necessary sanity checks.

    args = None
    if argv is not None:
        if default:
            warnings.warn('Provided "argv" will be ignored since "default" ' +
                          'option was turned on.')
        args = argv
    if default:
        args = []
    config = parser.parse_args(args=args)

    ### Check argument values!
    cli.check_invalid_argument_usage(config)
    common_args.check_invalid_args_general(config)

    return config


def main_net_args(agroup, allowed_nets=['mlp']):
    """This is a helper function for the function `parse_cmd_arguments` to add
    an argument group for options to a main network.

    Args:
        agroup (ArgumentGroup): An argument group to which the additional
            arguments are added.
        allowed_nets (list): List of allowed network identifiers. The following
            identifiers are considered (note, we also reference the network that
            each network type targets):

            - ``mlp``: :class:`mnets.mlp.MLP`
    """
    agroup.add_argument('--net_type', type=str, default=allowed_nets[0],
                        help='Type of network to be used for this ' +
                             'network. Default: %(default)s.',
                        choices=allowed_nets)

def init_args(agroup):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the initialization argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.init_args`.
    """
    agroup.add_argument('--keep_orig_init', action='store_true',
                        help='When converting the neural network into a ' +
                             'network with Gaussian weights, the main ' +
                             'network initialization (e.g., Xavier) will ' +
                             'be overwritten. This option assures that the ' +
                             'main network initialization is kept as an ' +
                             'initialization of the mean parameters in the ' +
                             'BNN.')


if __name__ == '__main__':
    pass

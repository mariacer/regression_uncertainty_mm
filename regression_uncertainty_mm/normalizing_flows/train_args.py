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
# @title          :normalizing_flows/train_args.py
# @author         :rd
# @contact        :rafael.daetwyler@uzh.ch
# @created        :11/24/2020
# @version        :1.0
# @python_version :3.7.4
"""
CLI Argument Parsing for normalizing flow experiments
-----------------------------------------------------

Command-line arguments and default values for the normalizing flow
experiments are handled here.
"""
import argparse
from datetime import datetime
import hypnettorch.utils.cli_args as cli
import warnings

import utils.train_args as common_args

def parse_cmd_arguments(mode='nf', dataset='toy_1D', default=False, argv=None):
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
        description = 'Simple 1D Regression - Normalizing Flow'
        dout_dir = './out_1D/run_' + curr_date
    elif dataset == 'toy_2D':
        description = 'Simple 2D Regression - Normalizing Flow'
        dout_dir = './out_2D/run_' + curr_date
    elif dataset == 'steering_angle':
        description = 'Steering angle prediction - Normalizing Flow'
        dout_dir = './out_steering_angle/run_' + curr_date
    else:
        raise ValueError('Dataset "%s" unknown.' % (dataset))

    if mode != 'nf':
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
                                synthetic_data=True, show_plots=True,
                                no_cuda=True, show_publication_style=True,
                                dout_dir=dout_dir)
        common_args.miscellaneous_args(misc_agroup, show_mean_only=True,
                                show_disable_lrt_test=True)
        miscellaneous_args(misc_agroup)
        # Evaluation arguments.
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                            show_val_set_size=True,
                                            dval_set_size=50)
        # Training arguments.
        train_argroup = cli.train_args(parser, show_lr=True, dn_iter=5000,
                       show_use_adam=True, show_use_rmsprop=True,
                       show_use_adadelta=True, show_use_adagrad=True,
                       show_epochs=True, show_clip_grad_value=True,
                       show_clip_grad_norm=True)
        common_args.train_args(train_argroup, show_local_reparam_trick=True,
                                         show_pred_dist_std=False,
                                         show_kl_scale=True)
        # Normalizing flow arguments.
        normalizing_flow_args(parser)
        # Initialization arguments.
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        common_args.mc_args(train_argroup, eval_agroup)
        # Network arguments.
        # For non-spline flows we'll have to add a main network here.
        agroup = cli.hnet_args(parser, show_cond_emb_size=False,
                               dhmlp_arch='10,10', show_specnorm=False,
                               show_batchnorm=False)

    elif dataset == 'toy_2D':
        ######################
        ### 2D Toy Dataset ###
        ######################
        common_args.twoD_regression_args(parser)
        # Miscellaneous arguments.
        misc_agroup = cli.miscellaneous_args(parser, big_data=False,
                                synthetic_data=True, show_plots=True,
                                no_cuda=True, show_publication_style=True,
                                dout_dir=dout_dir)
        common_args.miscellaneous_args(misc_agroup, show_mean_only=True,
                                show_disable_lrt_test=True)
        miscellaneous_args(misc_agroup)
        # Evaluation arguments.
        eval_agroup = cli.eval_args(parser, show_val_batch_size=True,
                                            show_val_set_size=True,
                                            dval_set_size=50)
        # Training arguments.
        train_argroup = cli.train_args(parser, show_lr=True, dn_iter=5000,
                       show_use_adam=True, show_use_rmsprop=True,
                       show_use_adadelta=True, show_use_adagrad=True,
                       show_epochs=True, show_clip_grad_value=True,
                       show_clip_grad_norm=True)
        common_args.train_args(train_argroup, show_local_reparam_trick=True,
                                         show_pred_dist_std=False,
                                         show_kl_scale=True)
        # Normalizing flow arguments.
        normalizing_flow_args(parser)
        # Initialization arguments.
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        common_args.mc_args(train_argroup, eval_agroup)
        # Network arguments.
        # For non-spline flows we'll have to add a main network here.
        agroup = cli.hnet_args(parser, show_cond_emb_size=False,
                               dhmlp_arch='10,10', show_specnorm=False,
                               show_batchnorm=False)

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
        miscellaneous_args(misc_agroup)
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
        common_args.train_args(train_agroup, show_local_reparam_trick=True,
                                     show_pred_dist_std=False,
                                     show_kl_scale=True)
        # Data arguments.
        cli.data_args(parser, show_data_dir=True,
                      ddata_dir='../datasets/udacity_ch2')
        # Normalizing flow arguments.
        normalizing_flow_args(parser)
        # Initialization arguments.
        cli.init_args(parser, custom_option=False, show_hyper_fan_init=True)
        common_args.mc_args(train_agroup, eval_agroup)
        # Network arguments.
        # Here the main and the hnet arguments all correspond to the
        # convolutional hypernetwork, with the main network corresponding to the
        # body and the hnet corresponding to the output head.
        # It is assumed that the flow is a spline, and not defined by the main
        # network arguments.
        mnet_group = cli.main_net_args(parser,
                    allowed_nets=['iresnet', 'lenet', 'resnet', 'wrn', 'mlp'],
                    show_no_bias=True, show_batchnorm=True,
                    show_no_batchnorm=False, show_bn_no_running_stats=True,
                    show_bn_distill_stats=False,
                    show_bn_no_stats_checkpointing=True,
                    show_specnorm=False, show_dropout_rate=True,
                    ddropout_rate=-1, show_net_act=True)
        hnet_group = cli.hnet_args(parser, show_cond_emb_size=False,
                               allowed_nets=['hmlp'], dhmlp_arch='',
                               show_specnorm=False,
                               show_batchnorm=False)
        convolutional_hnet_args(hnet_group)

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
    check_invalid_argument_usage(config, dataset)

    return config

def miscellaneous_args(agroup):
    """This is a helper function of the function :func:`parse_cmd_arguments` to
    add arguments to the miscellaneous argument group.

    Args:
        agroup: The argument group returned by method
            :func:`utils.cli_args.miscellaneous_args`.
    """
    agroup.add_argument('--ckpt_fn', type=str, default=None,
                        help='Path to a checkpoint of the hypernetwork. ' +
                             'Either this or ``--ckpt_path`` can be set.')
    agroup.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to a checkpoint of the hypernetwork. ' +
                             'The checkpoint with the best performance ' +
                             'measure will be chosen. Either this or ' +
                             '``--ckpt_fn`` can be set.')

def normalizing_flow_args(parser):
    """Adds flow specific arguments to the parser.

    Args:
        parser: Object of class :class:`argparse.ArgumentParser`.
    """
    agroup = parser.add_argument_group('Normalizing flow options')
    agroup.add_argument('--flow_depth', type=int, default=10, metavar='N',
                        help='Depth of the normalizing flow. ' +
                             'Default: %(default)s.')
    agroup.add_argument('--flow_layer_type', type=str, default='splines',
                        help='Type of layer used. Default: %(default)s.',
                        choices=['perceptron', 'splines'])

def convolutional_hnet_args(ngroup):
    """Adds arguments specific to convolutional hypernetworks.

    Args:
        ngroup: The argument group returned by method
            :func:`utils.cli_args.hnet_args`.
    """
    ngroup.add_argument('--hmlp_uncond_in_size', type=int, default=10,
                        help='The output size of the convolutional part of ' +
                             'the hypernetwork, and the input size to the ' +
                             'last MLP layer of the hypernetwork.')

def check_invalid_argument_usage(config, dataset):
    """Sanity check for normalizing flow command-line arguments.

    Args:
        config: Parsed command-line arguments.
    """
    if dataset == 'steering_angle':
        if config.flow_layer_type == 'perceptron':
            raise NotImplementedError


if __name__ == '__main__':
    pass

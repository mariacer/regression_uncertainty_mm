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
# @title          :utils/sim_utils.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :20/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Helper functions for simulations
--------------------------------

A collection of helper functions for simulations to keep other scripts clean.
"""
from argparse import Namespace
import numpy as np
import os
import sys
import torch

_SUMMARY_KEYWORDS = [
    # The weird prefix "aa_" makes sure keywords appear first in the result csv.
    'aa_val_loss_final',
    'aa_val_loss_lowest',
    'aa_val_loss_lowest_epoch',

    'aa_test_loss_final',
    'aa_test_loss_lowest',
    'aa_test_loss_at_lowest_val',

    # The following are only relevant for GL models.
    'aab_val_mse_final',
    'aab_val_mse_lowest',
    'aab_test_mse_final',
    'aab_test_mse_lowest',
    'aab_test_mse_at_lowest_val',
    'aab_train_mse_final',

    'num_nans',

    'num_weights_main',
    'num_weights_hyper',
    'num_weights_ratio',

    # Should be set in your program when the execution finished successfully.
    'finished'
]


def setup_summary_dict(config, shared, mnet, hnet=None):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).

    This method adds the keyword "summary" to `shared`.

    Args:
        config: Command-line arguments.
        shared: Miscellaneous data shared among training functions (summary dict
            will be added to this :class:`argparse.Namespace`).
        mnet: Main network.
        hnet (optional): Hypernetwork.
    """
    summary = dict()

    mnum = mnet.num_params

    hnum = -1
    hm_ratio = -1
    if hnet is not None:
        hnum = hnet.num_params
        hm_ratio = hnum / mnum

    # Note, we assume that all configs have the exact same keywords.
    summary_keys = _SUMMARY_KEYWORDS

    for k in summary_keys:
        if k in ['aa_val_loss_final', 'aa_val_loss_lowest',
                 'aa_val_loss_lowest_epoch',
                 'aa_test_loss_final', 'aa_test_loss_lowest',
                 'aa_test_loss_at_lowest_val', 'aab_test_mse_final',
                 'aab_val_mse_final', 'aab_val_mse_lowest',
                 'aab_test_mse_lowest', 'aab_test_mse_at_lowest_val',
                 'aab_train_mse_final']:
            summary[k] = -1
        elif k == 'num_nans':
            summary[k] = 0
        elif k == 'num_weights_main':
            summary[k] = mnum
        elif k == 'num_weights_hyper':
            summary[k] = hnum
        elif k == 'num_weights_ratio':
            summary[k] = hm_ratio
        elif k == 'finished':
            summary[k] = 0
        else:
            # Implementation must have changed if this exception is
            # raised.
            raise ValueError('Summary argument %s unknown!' % k)

    shared.summary = summary


def save_summary_dict(config, shared):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
    """
    # "setup_summary_dict" must be called first.
    assert(hasattr(shared, 'summary'))

    summary_fn = 'performance_overview.txt'
    #summary_fn = hpbbb._SUMMARY_FILENAME

    with open(os.path.join(config.out_dir, summary_fn), 'w') as f:
        for k, v in shared.summary.items():
            if isinstance(v, list):
                f.write('%s %s\n' % (k, utils.list_to_str(v)))
            elif isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            else:
                f.write('%s %d\n' % (k, v))


def backup_cli_command(config):
    """Write the curret CLI call into a script.

    This will make it very easy to reproduce a run, by just copying the call
    from the script in the output folder. However, this call might be ambiguous
    in case default values have changed. In contrast, all default values are
    backed up in the file ``config.json``.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
    """
    script_name = sys.argv[0]
    run_args = sys.argv[1:]
    command = 'python3 ' + script_name
    # FIXME Call reconstruction fails if user passed strings with white spaces.
    for arg in run_args:
        command += ' ' + arg

    fn_script = os.path.join(config.out_dir, 'cli_call.sh')

    with open(fn_script, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# The user invoked CLI call that caused the creation of\n')
        f.write('# this output folder.\n')
        f.write(command)

def setup_shared(config, device, mnet, hnet=None):
    """Set up simple structure used to share data among functions.

    Args:
        config: The experiment config.
        device: The device.
        mnet: The main network.
        hnet: The (optional) hypernetwork.

    Returns:
        (Namespace): The shared structure.
    """
    # Detect network where Bayesian stats might be applied.
    net = mnet
    if hnet is not None:
        net = hnet

    shared =  Namespace()

    # Mean and variance of prior that is used for variational inference.
    if config.mean_only: # No prior-matching can be performed.
        shared.prior_mean = None
        shared.prior_logvar = None
        shared.prior_std = None
    else:
        plogvar = np.log(config.prior_variance)
        pstd = np.sqrt(config.prior_variance)
        shared.prior_mean = [torch.zeros(*s).to(device) \
                             for s in net.orig_param_shapes]
        shared.prior_logvar = [plogvar * torch.ones(*s).to(device) \
                               for s in net.orig_param_shapes]
        shared.prior_std = [pstd * torch.ones(*s).to(device) \
                            for s in net.orig_param_shapes]

    return shared

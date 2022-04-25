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
# @title          :normalizing_flows/train_udacity.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :11/09/2021
# @version        :1.0
# @python_version :3.7.4
"""
Training script for a regression model based on normalizing flows
-----------------------------------------------------------------

This script trains a convolutional hypernetwork that learns to generate the
parameters of a predictive likelihood function parametrized by a normalizing
flow.
"""
# Do not delete the following import for all executable scripts!
import __init__  # pylint: disable=unused-import

from normalizing_flows.train import run

if __name__ == '__main__':
    run(mode='nf', dataset='steering_angle')


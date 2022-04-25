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
# @title          :gaussian_likelihoods/train_ghl.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :16/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Training script for Gaussian likelihood (heteroscedastic) regression models
---------------------------------------------------------------------------
"""
# Do not delete the following import for all executable scripts!
import __init__  # pylint: disable=unused-import

from gaussian_likelihoods.train import run

if __name__ == '__main__':
    _ = run(mode='ghl')


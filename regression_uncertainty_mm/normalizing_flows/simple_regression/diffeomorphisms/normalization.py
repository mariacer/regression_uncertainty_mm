#!/usr/bin/env python3
# Copyright 2021 Maria Cervera

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
# @title          :normalizing_flows/simple_regression/diffeomorphisms/normalization.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :24/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Normalization diffeomorphism
----------------------------

Original code by Francesco d'Angelo and Rafael Daetwyler. 
"""
import torch
import numpy as np

from normalizing_flows.simple_regression.diffeomorphisms.diffeom_interface \
    import Diffeomorphism

class Normalization(Diffeomorphism):

    @staticmethod
    def forward(inputs, **kwargs):
        assert type(kwargs['scale']) in (int, float, np.double)
        assert type(kwargs['translation']) in (int, float, np.double)
        assert kwargs['scale'] != 0

        return (inputs-kwargs['translation'])/kwargs['scale']

    @staticmethod
    def forward_derivative(inputs, **kwargs):
        assert type(kwargs['scale']) in (int, float, np.double)
        assert kwargs['scale'] != 0

        return torch.ones_like(inputs)/kwargs['scale']

    @staticmethod
    def inverse(inputs, **kwargs):
        assert type(kwargs['scale']) in (int, float, np.double)
        assert type(kwargs['translation']) in (int, float, np.double)
        assert kwargs['scale'] != 0

        return inputs*kwargs['scale']+kwargs['translation']

    @staticmethod
    def inverse_derivative(inputs, **kwargs):
        assert type(kwargs['scale']) in (int, float, np.double)
        assert kwargs['scale'] != 0

        return torch.ones_like(inputs)*kwargs['scale']

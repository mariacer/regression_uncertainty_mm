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
# @title          :normalizing_flows/simple_regression/diffeomorphisms/leaky_relu.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :24/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Leaky relu diffeomorphism
-------------------------

Original code by Francesco d'Angelo and Rafael Daetwyler. 
"""
import torch
import torch.nn.functional as F

from normalizing_flows.simple_regression.diffeomorphisms.diffeom_interface \
    import Diffeomorphism

class LeakyReLU(Diffeomorphism):
    """Implementation of a leaky relu diffeomorphism.
    """
    @staticmethod
    def forward(inputs, **kwargs):
        assert type(kwargs['negative_slope']) in (int, float)
        assert kwargs['negative_slope'] > 0

        return F.leaky_relu(inputs, kwargs['negative_slope'])

    @staticmethod
    def forward_derivative(inputs, **kwargs):
        assert type(kwargs['negative_slope']) in (int, float)
        assert kwargs['negative_slope'] > 0

        return ((inputs >= 0).type(torch.FloatTensor) +
               (inputs < 0).type(torch.FloatTensor)*kwargs['negative_slope']).to(inputs.device)

    @staticmethod
    def inverse(inputs, **kwargs):
        assert type(kwargs['negative_slope']) in (int, float)
        assert kwargs['negative_slope'] > 0

        return F.leaky_relu(inputs, 1/kwargs['negative_slope'])

    @staticmethod
    def inverse_derivative(inputs, **kwargs):
        assert type(kwargs['negative_slope']) in (int, float)
        assert kwargs['negative_slope'] > 0

        return LeakyReLU.forward_derivative(inputs, negative_slope=1/kwargs['negative_slope'])

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
# @title          :normalizing_flows/simple_regression/diffeomorphisms/log_transform.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :24/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Log transform diffeomorphism
-------------------------

Original code by Francesco d'Angelo and Rafael Daetwyler. 
"""
import torch

from normalizing_flows.simple_regression.diffeomorphisms.diffeom_interface \
    import Diffeomorphism

class LogTransform(Diffeomorphism):

    @staticmethod
    def forward(inputs, **kwargs):
        pos = (inputs >= 0).type(torch.FloatTensor).to(inputs.device) * torch.log(inputs + 1)
        pos[torch.isnan(pos)] = 0
        neg = (inputs < 0).type(torch.FloatTensor).to(inputs.device)  * torch.log(-inputs + 1)
        neg[torch.isnan(neg)] = 0
        return pos - neg

    @staticmethod
    def forward_derivative(inputs, **kwargs):
        pos = (inputs >= 0).type(torch.FloatTensor).to(inputs.device) / (inputs + 1)
        pos[torch.isnan(pos)] = 0
        neg = (inputs < 0).type(torch.FloatTensor).to(inputs.device) / (inputs - 1)
        neg[torch.isnan(neg)] = 0
        return pos - neg

    @staticmethod
    def inverse(inputs, **kwargs):
        return ((inputs >= 0).type(torch.FloatTensor).to(inputs.device) * (torch.exp(inputs) - 1) -
                (inputs < 0).type(torch.FloatTensor).to(inputs.device)  * (torch.exp(-inputs) - 1))

    @staticmethod
    def inverse_derivative(inputs, **kwargs):
        return ((inputs >= 0).type(torch.FloatTensor).to(inputs.device) * torch.exp(inputs) +
                (inputs < 0).type(torch.FloatTensor).to(inputs.device)  * torch.exp(-inputs))

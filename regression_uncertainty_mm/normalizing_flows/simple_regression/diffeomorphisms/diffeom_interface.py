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
# @title          :normalizing_flows/simple_regression/diffeomorphisms/diffeomorphism_interface.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :24/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Main-Network Interface
----------------------

The module :mod:`diffeomorphism_interface` contains an interface
diffeomorphisms. The interface ensures that we can consistently use these
functions without knowing their specific implementation.

Original code by Francesco d'Angelo and Rafael Daetwyler. 
"""

from abc import ABC, abstractmethod

class Diffeomorphism(ABC):
    """A general interface for diffeomorphisms for normalizing flows.

    Attributes:
        internal_params (torch.nn.ParameterList or None): A list of all
    """

    @abstractmethod
    def forward(inputs, **kwargs):
        """Forward pass through the flow."""
        raise NotImplementedError()

    @abstractmethod
    def forward_derivative(inputs, **kwargs):
        """Derivative of the forward pass through the flow."""
        raise NotImplementedError()

    @abstractmethod
    def inverse(inputs, **kwargs):
        """Inverse pass through the flow."""
        raise NotImplementedError()

    @abstractmethod
    def inverse_derivative(inputs, **kwargs):
        """Derivative of the inverse pass through the flow."""
        raise NotImplementedError()

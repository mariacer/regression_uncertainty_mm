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
# @title          :normalizing_flows/simple_regression/layers/perceptron.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :24/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Perceptron for a normalizing flow
---------------------------------

Original code by Francesco d'Angelo and Rafael Daetwyler. 
"""
from nflows.transforms.base import Transform
import torch

class Perceptron(Transform):
    """Implementation of a perceptron layer for normalizing flows.
    """

    def forward(self, inputs, context):
        """Forward pass through the layer.

        Args:
            input (torch.Tensor): The inputs to the layer.
            context (list): Miscellaneous information. It is a list of length
                four containing:

                * the parameters (weights and biases) of the
                * the index of the layer
                * the nonlinearity
                * a dictionary of remaining kwargs 

        Returns:
            (tuple): Tuple containing:

            - **output**: The output of the layer.
            - **logabsdet**: The log of the absolute values of the derivative.
        """
        (w_weights, b_weights), i, act_fn, kwargs = context

        # Compute the forward pass applying the non-linearity.
        lin = w_weights[i] * inputs + b_weights[i]
        output = act_fn.forward(lin, **kwargs)

        # Compute the derivative of the forward pass.
        f_deriv = w_weights[i] * act_fn.forward_derivative(lin, **kwargs)

        # Compute the log of the absolute value of the derivative.
        logabsdet = torch.log(torch.abs(f_deriv))  # +1e-7

        return output, logabsdet

    def inverse(self, inputs, context):
        """Inverse pass through the layer.

        Args:
            input (torch.Tensor): The (inverse) inputs to the layer.
            (....): See docstring of function :func:`forward`.

        Returns:
            (tuple): Tuple containing:

            - **output**: The output of the layer.
            - **logabsdet**: The log of the absolute values of the derivative.
        """
        (w_weights, b_weights), i, act_fn, kwargs = context

        # Compute the inverse pass applying the inverse of the non-linearity.
        output = (act_fn.inverse(inputs, **kwargs) - b_weights[i]) / w_weights[i]

        # Compute the derivative of the forward pass (bias terms drops).
        deriv = act_fn.inverse_derivative(inputs, **kwargs) / w_weights[i]

        # Compute the log of the absolute value of the derivative.
        logabsdet = torch.log(torch.abs(deriv))  # +1e-7
        
        return output, logabsdet

    @staticmethod
    def preprocess_weights(weights, bias):
        """Preprocess the weights.

        Args:
            weights (torch.Tensor): The weights.
            bias (bool): Whether a bias term exists.

        Returns:
            (tuple): Tuple containing:

            -**w_weights**: The weights.
            -**b_weights**: The biases.
        """
        # TODO: adapt this for hnet.forward(ret_format='flattened') output
        device = weights[0][0].device
        batch_size = len(weights)
        depth = len(weights[0]) // 2 if bias else len(weights[0])
        w_weights = [torch.empty(batch_size, device=device) for _ in \
            range(depth)]

        if bias:
            b_weights = [torch.empty(batch_size, device=device) for _ in \
                range(depth)]
            for i, p in enumerate(weights):
                assert (len(p) // 2 == depth)
                for j, o in enumerate(p):
                    if j % 2 == 1:
                        b_weights[j // 2][i] = o
                    else:
                        w_weights[j // 2][i] = Perceptron.make_invertible(o)
        else:
            b_weights = [torch.zeros(batch_size, device=device) for _ \
                in range(depth)]
            for i, p in enumerate(weights):
                assert (len(p) == depth)
                for j, o in enumerate(p):
                    w_weights[j][i] = Perceptron.make_invertible(o)

        return w_weights, b_weights

    @staticmethod
    def make_invertible(w, eps=1e-6):
        """Make weight matrix invetible.

        Ensures that all weight magnitues are at least `eps`.

        Args:
            w: The weight matrix.
            eps (float): The value to be added to the weights.

        Returns:
            The weights.
        """
        return w + ((w >= 0) and (w < eps))*eps \
                 - ((w < 0) and (w > -eps))*eps

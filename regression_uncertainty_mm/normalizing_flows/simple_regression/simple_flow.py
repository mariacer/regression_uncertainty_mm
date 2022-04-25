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
# @title          :normalizing_flows/simple_regression/simple_flow.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :24/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Simple normalizing flow for 1D regression experiments
-----------------------------------------------------

Original code by Francesco d'Angelo and Rafael Daetwyler. 
"""
from hypnettorch.mnets.mlp import MLP
import torch
import torch.nn as nn

from normalizing_flows.simple_regression.diffeomorphisms.leaky_relu import \
    LeakyReLU
from normalizing_flows.simple_regression.diffeomorphisms.log_transform import \
    LogTransform
from normalizing_flows.simple_regression.layers.perceptron import Perceptron
from normalizing_flows.simple_regression.layers.splines import \
    PiecewiseRationalQuadraticCDF

class SimpleFlow(MLP):
    r"""Implementation of an MLP-based normalizing flow.

    This is a normalizing flow based on a simple fully-connected network, that
    receives input vector :math:`\mathbf{t}` and outputs a vector
    :math:`\mathbf{u}` of real values.

    Args:
        depth (int): The depth (number of layers) of the flow.
        dimensionality (int): The dimensionality of the inputs and outputs.
        layers (str): The type of layer to be used.
        activation_fn: The nonlinearity used in hidden layers. If ``None``, no
            nonlinearity will be applied.
        use_bias (bool): Whether layers may have bias terms.
        in_fn (optional): If provided, this function will be applied to the
            input neurons of the network.
        out_fn (optional): If provided, this function will be applied to the
            output neurons of the network.

            Warning:
                This changes the interpretation of the output of the
                :meth:`forward` method.
    """
    def __init__(self, depth, dimensionality=1, layers='perceptron',
                 activation_fn=LeakyReLU, use_bias=True, in_fn=None,
                 out_fn=LogTransform, spline_bound=1.0, **kwargs):
        # TODO: inherit directly from MainNetInterface instead of MLP,
        # as we don't need most of the stuff from MLP

        # Note, I suspect this activation_fn input isn't used, as the forward
        # method is overwritten.
        MLP.__init__(self, hidden_layers=[1]*depth, activation_fn=activation_fn,
                     use_bias=use_bias, no_weights=True, out_fn=out_fn)

        if dimensionality != 1:
            raise NotImplementedError

        self._depth = depth
        self._kwargs = kwargs
        self._in_fn = in_fn

        if layers == 'splines':
            self._layer = PiecewiseRationalQuadraticCDF(shape=dimensionality,
                                                        tails='linear',
                                                        tail_bound=spline_bound)
            # TODO: other properties besides param_shapes should also be changed
            # also, think of changing the parent class
            single_shapes = self._layer.get_param_shapes()
            self._param_shapes = [shape for _ in range(self._depth) \
                                  for shape in single_shapes]
            self._param_shapes_meta = [\
                                    {'name': 'weight', 'index': -1, 'layer': i}
                                    for i in range(self._depth)
                                    for _ in single_shapes]
            self._hyper_shapes_learned = self._param_shapes
            self._a_fun = None
        elif layers == 'perceptron':
            self._layer = Perceptron()
        else:
            raise NotImplementedError('Layer type '+layers+' not implemented!')

        # Define the base distribution p(u).
        self._p0 = torch.distributions.normal.Normal(torch.tensor([0.0]), \
                                                     torch.tensor([1.0]))

    def forward(self, t, weights, preprocessed=False):
        """Forward pass through the flow.

        Args:
            t (torch.Tensor): The inputs to the flow.
            weights (torch.Tensor): The weights of the flow.
            preprocessed: Whether the weights have been preprocessed.

        Returns:
            (tuple): Tuple containing:

            - **u**: The output of the flow.
            - **logabsdet**: The absolute value of the log of the Jacobian of
                the flow.
        """
        if not preprocessed:
            assert(len(weights) == len(t))
            weights = self._layer.preprocess_weights(weights)

        hidden = torch.squeeze(t)

        # input layer
        if self._in_fn is not None:
            context = [weights, -1, self._in_fn, self._kwargs]
            hidden, logabsdet = self._layer.forward(hidden, context)

        # hidden layers
        for i in range(self._depth):
            context = [weights, i, self._a_fun, self._kwargs]
            hidden, temp = self._layer.forward(hidden, context)
            logabsdet = logabsdet + temp if 'logabsdet' in locals() else temp

        # output layer
        if self._out_fn is not None:
            context = [weights, self._depth, self._out_fn, self._kwargs]
            hidden, temp = self._layer.forward(hidden, context)
            logabsdet += temp

        u = hidden

        return u, logabsdet

    def inverse(self, u, weights, preprocessed=False):
        """Inverse pass through the flow.

        Args:
            (....): See docstring of method :func:`forward`.

        Returns:
            (....): See docstring of method :func:`forward`.
        """
        if not preprocessed:
            assert(len(weights) == len(u))
            weights = self._layer.preprocess_weights(weights)

        hidden = torch.squeeze(u)

        # output layer
        if self._out_fn is not None:
            context = [weights, self._depth, self._out_fn, self._kwargs]
            hidden, logabsdet = self._layer.inverse(hidden, context)

        # hidden layers
        for i in reversed(range(self._depth)):
            context = [weights, i, self._a_fun, self._kwargs]
            hidden, temp = self._layer.inverse(hidden, context)
            logabsdet = logabsdet + temp if 'logabsdet' in locals() else temp

        # input layer
        if self._in_fn is not None:
            context = [weights, -1, self._in_fn, self._kwargs]
            hidden, temp = self._layer.inverse(hidden, context)
            logabsdet += temp

        y = hidden

        return y, logabsdet

    def log_p0(self, u):
        """Compute the log of the probability of the input.

        Args:
            u (torch.Tensor): The sample for which to evaluate the probability.

        Returns:
            (float): The probability.
        """
        return self._p0.log_prob(u)

    def sample_p0(self, n):
        """Same from the base distribution.

        Args:
            n (int): The number of samples to get.

        Returns:
            (torch.Tensor): The samples.
        """
        if isinstance(n, int):
            n = torch.Size([n])
        return self._p0.sample(n)

    def get_max_deviation(self, x, weights):
        """Get maximum deviation between forward and backward passes.

        Args:
            x (torch.Tensor): The forward inputs.
            weights: The weights of the layer.

        Returns:
            (float): The maximum absolute deviation across the batch.
        """
        forwardPass, _ = self.forward(x, weights)
        inversePass, _ = self.inverse(forwardPass, weights)

        return torch.max(torch.abs(x.squeeze() - inversePass)).item()


    def to(self, *args, **kwargs):
        """Move model to specific device."""
        nn.Module.to(self, *args, **kwargs)

        if 'device' in kwargs:
            device = kwargs['device']
        elif isinstance(args[0], torch.device):
            device = args[0]
        else:
            raise NotImplementedError

        loc = self._p0.loc
        scale = self._p0.scale
        self._p0 = torch.distributions.normal.Normal(loc.to(device), \
                                                     scale.to(device))
        return self

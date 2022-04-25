#!/usr/bin/env python3
# Copyright 2021 Hamza Keurti

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
# @title          :normalizing_flows/simple_regression/bivariate_flow.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :13/09/2021
# @version        :1.0
# @python_version :3.7.4
"""
Bivariate normalizing flow for 2D regression experiments
--------------------------------------------------------

Original code by Hamza Keurti. 
"""
import numpy as np
from hypnettorch.hnets import HMLP
from hypnettorch.mnets.mlp import MLP
import hypnettorch.utils.misc as misc
import torch
import torch.nn as nn

from normalizing_flows.simple_regression.diffeomorphisms.leaky_relu import \
    LeakyReLU
from normalizing_flows.simple_regression.diffeomorphisms.log_transform import \
    LogTransform
from normalizing_flows.simple_regression.layers.perceptron import Perceptron
from normalizing_flows.simple_regression.layers.splines import \
    PiecewiseRationalQuadraticCDF


class BivariateFlow(MLP):
    r"""Implementation of a multi-dimensional normalizing flow.

    This is a normalizing flow based on our simple 1-D version, but that allows
    having outputs with higher dimensions.

    Args:
        (....): See docstring of class :class:`simple_flow`.
    """
    def __init__(self, depth, permute=None, layers='perceptron',
                 conditioner_arch='10,10',
                 activation_fn=LeakyReLU, use_bias=True, in_fn=None,
                 out_fn=LogTransform, spline_bound=1.0, **kwargs):
        MLP.__init__(self, hidden_layers=[1]*depth, activation_fn=activation_fn,
                     use_bias=use_bias, no_weights=True, out_fn=out_fn)        
                     
        self._depth = depth
        self._kwargs = kwargs
        self._in_fn = in_fn
        if permute is None or len(permute) == 0:
            permute = [i%2 for i in range(depth)]
        assert len(permute) == depth

        self._permute = permute

        # Transformer
        if layers == 'splines':
            self._transformer = PiecewiseRationalQuadraticCDF(shape=1,
                                                        tails='linear',
                                                        tail_bound=spline_bound)

            transformer_shapes = self._transformer.get_param_shapes()
            self._hyper_shapes_learned = self._param_shapes
            self._a_fun = None
        elif layers == 'perceptron':
            self._transformer = Perceptron()
        else:
            raise NotImplementedError('Layer type '+layers+' not implemented!')

        # Conditionner takes one factor of the input distribution as input and 
        # outputs weights for the transformer of the other factor. 
        # Conditioner maintains no internal weights, 
        # they are provided by the hnet. 
        self._conditioner = HMLP(target_shapes=transformer_shapes,
                uncond_in_size=1, cond_in_size=0, no_cond_weights=True, 
                no_uncond_weights=True, num_cond_embs=0, 
                layers=conditioner_arch)
        single_shapes = self._conditioner._param_shapes
        self._param_shapes = [shape for _ in range(self._depth) \
                                  for shape in single_shapes]
        self._param_shapes_meta = [\
                        {'name': 'weight', 'index': -1, 'layer':i}
                        for i in range(self._depth)
                        for _ in single_shapes]
        
        # Define the base distribution p(u). 
        self._p0 = torch.distributions.multivariate_normal.MultivariateNormal(
                                                     torch.tensor([0.0, 0.0]),
                                                     torch.tensor([[1.0,0.0],
                                                                   [0.0,1.0]]))


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
        batch_size = t.shape[0]
        if not preprocessed:
            assert(len(weights) == len(t))
            weights = self._preprocess_weights(weights)

        hidden = torch.squeeze(t)

        # input layer
        if self._in_fn is not None:
            context = [None, -1, self._in_fn, self._kwargs]
            hidden, logabsdet = self._transformer.forward(hidden, context)
            # Jacobian of activations is diagonal, determinant is product.
            logabsdet = logabsdet.sum(axis = 1)

        # hidden layers
        for i in range(self._depth):
            if self._permute[i]:
                # Whether to permute the two factors.
                hidden = hidden.flip((-1,))
            transformer_weights = []
            for b in range(batch_size):
                w = self._conditioner.forward(hidden[b,0].view(1,1),
                                              weights = weights[i][b])
                transformer_weights.append(w)
            transformer_weights = [
                    torch.cat(
                        [w[k] for w in transformer_weights]) for k in range(3)]
            context = [transformer_weights, 0, self._a_fun, self._kwargs]
            hidden_temp, temp = self._transformer.forward(\
                hidden[...,1], context, single_layer=True)
            hidden = torch.stack((hidden[:,0],hidden_temp),axis=1)
            logabsdet = logabsdet + temp if 'logabsdet' in locals() else temp

        # output layer
        # FIXME logabsdet. 
        # Expected shape: [n_samples]
        if self._out_fn is not None:
            context = [None, self._depth, self._out_fn, self._kwargs]
            hidden, temp = self._transformer.forward(hidden, context)
            temp = temp.sum(axis = 1)
            logabsdet += temp

        u = hidden

        return u, logabsdet

    def _preprocess_weights(self, flat_weights):
        """Helper function to convert flat weights into target net shape.
        Adapted from hypnettorch.hnets.hnet_interface.py

        Args:
            flat_weights (torch.Tensor): The flat output tensor corresponding to
                ``ret_format='flattened'``.

        Returns:
            (list or torch.)
        """
        assert len(flat_weights.shape) == 2
        batch_size = flat_weights.shape[0]
        single_shapes = self._conditioner._param_shapes
        num_layer = self._conditioner._num_params
        ind_l = 0
        ret = [[[] for _ in range(batch_size)] 
                   for __ in range(self._depth)]
        for l in range(self._depth):            
            ind = 0
            layer_W = flat_weights[:, ind_l:ind_l+num_layer]
            for s in single_shapes:
                num = int(np.prod(s))

                W = layer_W[:, ind:ind+num]
                W = W.view(batch_size, *s)

                for bind, W_b in enumerate(torch.split(W, 1, dim=0)):
                    W_b = torch.squeeze(W_b, dim=0)
                    assert np.all(np.equal(W_b.shape, s))
                    ret[l][bind].append(W_b)

                ind += num
            ind_l += num_layer
        return ret
    
    def inverse(self, u, weights, preprocessed=False):
        """Inverse pass through the flow.

        Args:
            (....): See docstring of method :func:`forward`.

        Returns:
            (....): See docstring of method :func:`forward`.
        """
        pass


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
        covariance_matrix = self._p0.covariance_matrix
        self._p0 = torch.distributions.multivariate_normal.MultivariateNormal(
                                            loc.to(device),
                                            covariance_matrix.to(device))
        return self

    def log_p0(self, u):
        """Compute the log of the probability of the input.

        Args:
            u (torch.Tensor): The sample for which to evaluate the probability.

        Returns:
            (float): The probability.
        """
        return self._p0.log_prob(u)
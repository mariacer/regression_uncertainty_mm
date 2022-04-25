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
# @title          :normalizing_flows/simple_regression/layers/splines.py
# @author         :mc
# @contact        :mariacer@ethz.ch
# @created        :24/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
Spline layer to be used for normalizing flows
---------------------------------------------

Original code by Francesco d'Angelo and Rafael Daetwyler, adapted from
https://github.com/bayesiains/nflows.
"""
from nflows.transforms import splines
from nflows.transforms.base import Transform
import torch

# TODO: consider inheriting nn.Module directly
class PiecewiseRationalQuadraticCDF(Transform):
    r"""Implementation of a piecewise rational quadratic CDF.

    This class can be used as a layer for a normalizing flow.

    Args:
        shape (int): The size of the inputs and outputs.
        num_bins (int): The number of spline segments.
        tails (str or None): The type of tails used. If ``linear``, no
            derivatives will be used at the tails.
        max_bin_width (float): The maximum bin width.
        min_bin_width (float): The minimum bin width.
        min_derivative (float): The minimum derivative in the segments.
    """
    def __init__(self, shape, num_bins=10, tails=None, tail_bound=1.0,
        min_bin_width=splines.rational_quadratic.DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=splines.rational_quadratic.DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=splines.rational_quadratic.DEFAULT_MIN_DERIVATIVE):
        super().__init__()

        self.num_bins = num_bins
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        self.tail_bound = tail_bound
        self.tails = tails
        self.num_derivatives = (num_bins - 1) if tails == "linear" else \
                               (num_bins + 1)
        if isinstance(shape, int):
            shape = (shape,)
        self.in_shape = shape

    def _spline(self, inputs, unnormalized_widths, unnormalized_heights,
                unnormalized_derivatives, inverse=False):
        """The actual spline computation.

        Args:
            inputs (torch.Tensor): The inputs.
            unnormalized_widths (torch.Tensor): Widths of the spline segments.
            unnormalized_heights (torch.Tensor): Heights of the spline segments.
            unnormalized_derivatives (torch.Tensor): Derivatives of the spline
                segments.
            inverse (bool): Whether this is the inverse transformation.
            (....): See dosctring of class
                :class:`PiecewiseRationalQuadraticCDF`.

        Returns:
            (tuple): Tuple containing:

            - **outputs**: The outputs of the layer.
            - **logabsdet**: The absolute value of the log of the determinant
                of the Jacobian of the spline.
        """
        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        outputs, logabsdet = spline_fn(inputs=inputs,
                            unnormalized_widths=unnormalized_widths,
                            unnormalized_heights=unnormalized_heights,
                            unnormalized_derivatives=unnormalized_derivatives,
                            inverse=inverse,
                            min_bin_width=self.min_bin_width,
                            min_bin_height=self.min_bin_height,
                            min_derivative=self.min_derivative,
                            **spline_kwargs)

        return outputs, logabsdet

    def forward(self, inputs, context=None, single_layer = False):
        """Forward pass through the spline.

        Args:
            inputs (torch.Tensor): The inputs.
            context (list): Miscellaneous information. It is a list of length
                four containing:

                * the parameters (weights and biases) of the
                * the index of the layer
                * the nonlinearity
                * a dictionary of remaining kwargs 
            single_layer (bool): Whether the context corresponds to a 
                single layer or all layers.

        Returns:
            (torch.Tensor): The output.
        """
        weights, i, act_fn, kwargs = context

        # If act_fn is set, splines will be ignored and the input will directly
        # be passed through act_fn.
        if act_fn is not None:
            output = act_fn.forward(inputs, **kwargs)
            f_deriv = act_fn.forward_derivative(inputs, **kwargs)
            logabsdet = torch.log(torch.abs(f_deriv))  # +1e-7
            return output, logabsdet

        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = \
            weights
        
        if single_layer:
            # context contains weights for one layer at a time
            return self._spline(inputs, unnormalized_widths,
                            unnormalized_heights,
                            unnormalized_derivatives, inverse=False)


        return self._spline(inputs, unnormalized_widths[i],
                            unnormalized_heights[i],
                            unnormalized_derivatives[i], inverse=False)

    def inverse(self, inputs, context=None):
        """Inverse pass through the spline.

        Args:
            inputs (torch.Tensor): The inputs.
            context (list): The context for the computation.

        Returns:
            (torch.Tensor): The output.
        """
        weights, i, act_fn, kwargs = context

        # If act_fn is set, splines will be ignored and the input will directly
        # be passed through act_fn.
        if act_fn is not None:
            output = act_fn.inverse(inputs, **kwargs)
            f_deriv = act_fn.inverse_derivative(inputs, **kwargs)
            logabsdet = torch.log(torch.abs(f_deriv))  # +1e-7
            return output, logabsdet

        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = \
            weights
        return self._spline(inputs, unnormalized_widths[i],
                            unnormalized_heights[i],
                            unnormalized_derivatives[i], inverse=True)

    def preprocess_weights(self, weights):
        """Preprocess the weights.

        Args:
            weights (torch.Tensor): The weights.

        Returns:
            (tuple): Tuple containing:

            -**uw**: The unnormalized widths.
            -**uh**: The unnormalized heights.
            -**ud**: The unnormalized derivatives.
        """
        mod_s = 2*self.num_bins + self.num_derivatives
        nb = self.num_bins
        nd = self.num_derivatives
        depth = weights.shape[1] // mod_s
        uw = []
        uh = []
        ud = []
        for i in range(depth):
            uw.append(weights[:, i*mod_s:i*mod_s+nb])
            uh.append(weights[:, i*mod_s+nb:i*mod_s+2*nb])
            ud.append(weights[:, i*mod_s+2*nb:i*mod_s+2*nb+nd])

        return uw, uh, ud

    def get_param_shapes(self):
        """Get the shapes of the trainable parameters.

        Returns:
            (list): The unnormalized widths, heights and derivatives at each
                spline segment.
        """
        return [[*self.in_shape, self.num_bins],         # unnormalized_widths
                [*self.in_shape, self.num_bins],         # unnormalized_heights
                [*self.in_shape, self.num_derivatives]]  # unnormalized_derivs

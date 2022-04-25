#!/usr/bin/env python3
# Copyright 2021 Christian Henning
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
# @title          :networks/mnet_hnet_wrapper.py
# @author         :ch
# @contact        :henningc@ethz.ch
# @created        :09/10/2021
# @version        :1.0
# @python_version :3.6.9
"""
Combining a main-network's features with a hypernetwork
-------------------------------------------------------

This wrapper is meant to take an existing main-network instance and feed it's
output into a given hypernetwork instance, which then can use those expressive
features to generate the parameters of an auxiliary network.

Note, in general, each hypernetwork should implement the interface
:class:`hypnettorch.hnets.hnet_interface.HyperNetInterface`. However, for fast
prototyping it might be convinient to sometimes combine existing feature
extractors that implement
:class:`hypnettorch.mnets.mnet_interface.MainNetInterface` with existing
hypernets. This wrapper provides this capability, but it should be used with
care!

The wrapper will not implement the interface
:class:`hypnettorch.hnets.hnet_interface.HyperNetInterface` and therefore does
not provide full hypernetwork capabilities!
"""
from hypnettorch.hnets.hnet_interface import HyperNetInterface
from hypnettorch.mnets.mnet_interface import MainNetInterface
from torch import nn

class MnetHnetWrapper(nn.Module, MainNetInterface):
    """Wrapper to combine an existing main and hypernetwork.

    The networks must have been instantiated, such that the outputs of the
    main network can simply be fed into the hypernetwork.

    Args:
        mnet (hypnettorch.mnets.mnet_interface.MainNetInterface): Existing
            main network instance (feature extractor).
        hnet (hypnettorch.hnets.hnet_interface.HyperNetInterface): Existing
            hypernetwork instance (weight generator).
    """
    def __init__(self, mnet, hnet):
        # FIXME find a way using super to handle multiple inheritance.
        nn.Module.__init__(self)
        MainNetInterface.__init__(self)

        assert isinstance(mnet, MainNetInterface)
        assert isinstance(hnet, HyperNetInterface)

        self._mnet = mnet
        self._hnet = hnet

        # Setup required attributes!
        if mnet.internal_params is None:
            assert hnet.internal_params is None
        else:
            assert hnet.internal_params is not None
            self._internal_params = nn.ParameterList()
            self._internal_params.extend(mnet.internal_params)
            self._internal_params.extend(hnet.internal_params)

        self._param_shapes = mnet.param_shapes + hnet.param_shapes
        # You don't have to implement this following attribute, but it might
        # be helpful, for instance for hypernetwork initialization.
        self._param_shapes_meta = mnet.param_shapes_meta + \
            hnet.param_shapes_meta

        if mnet.hyper_shapes_learned is None:
            assert hnet.hyper_shapes_learned is None or \
                len(hnet.hyper_shapes_learned) == 0
        else:
            assert hnet.hyper_shapes_learned is not None
            self._hyper_shapes_learned = mnet.hyper_shapes_learned + \
                hnet.hyper_shapes_learned

        if mnet.hyper_shapes_learned_ref is None:
            assert hnet.hyper_shapes_learned_ref is None or \
                len(hnet.hyper_shapes_learned_ref) == 0
        else:
            assert hnet.hyper_shapes_learned_ref is not None
            self._hyper_shapes_learned_ref = mnet.hyper_shapes_learned_ref + \
                hnet.hyper_shapes_learned_ref

        if mnet.hyper_shapes_distilled is None:
            assert hnet.hyper_shapes_distilled is None
        else:
            assert hnet.hyper_shapes_distilled is not None
            self._hyper_shapes_distilled = mnet.hyper_shapes_distilled + \
                hnet.hyper_shapes_distilled

        try:
            self._has_bias = None
            if mnet.has_bias and hnet.has_bias:
                self._has_bias = True
            elif not mnet.has_bias and not hnet.has_bias:
                self._has_bias = False
        except:
            self._has_bias = None

        self._has_fc_out = hnet.has_fc_out
        self._mask_fc_out = hnet.mask_fc_out
        self._has_linear_out = hnet.has_linear_out

        self._layer_weight_tensors = nn.ParameterList()
        self._layer_weight_tensors.extend(mnet.layer_weight_tensors)
        self._layer_weight_tensors.extend(hnet.layer_weight_tensors)

        self._layer_bias_vectors = nn.ParameterList()
        self._layer_bias_vectors.extend(mnet.layer_bias_vectors)
        self._layer_bias_vectors.extend(hnet.layer_bias_vectors)

        self._batchnorm_layers = mnet.batchnorm_layers
        if hnet.batchnorm_layers is not None:
            if self._batchnorm_layers is None:
                self._batchnorm_layers = hnet.batchnorm_layers
            else:
                self._batchnorm_layers.extend(hnet.batchnorm_layers)

        self._context_mod_layers = mnet.context_mod_layers
        if hnet.context_mod_layers is not None:
            if self._context_mod_layers is None:
                self._context_mod_layers = hnet.context_mod_layers
            else:
                self._context_mod_layers.extend(hnet.context_mod_layers)

        self._is_properly_setup(check_has_bias=False)

    @property
    def internal_hnet(self):
        """The internally maintained hypernetwork.

        See contructor argument ``hnet``.

        :type: hypnettorch.hnets.hnet_interface.HyperNetInterface
        """
        return self._hnet

    @property
    def internal_mnet(self):
        """The internally maintained main network.

        See contructor argument ``mnet``.

        :type: hypnettorch.mnets.mnet_interface.MainNetInterface
        """
        return self._mnet

    @property
    def has_bias(self):
        """Getter for read-only attribute :attr:`has_bias`."""
        if self._has_bias is None:
            raise RuntimeError('Attribute "has_bias" does not apply.')
        return self._has_bias

    def distillation_targets(self):
        """Targets to be distilled after training."""
        raise NotImplementedError()

    def forward(self, x, weights=None, distilled_params=None, condition=None,
                **kwargs):
        """Compute network output.

        This function passes the input ``x`` throught the internal main network
        and then passes the corresponding output through the internal hypernet.

        Args:
            (....) See docstring of method
                :meth:`hypnettorch.mnets.mnet_interface.MainNetInterface.\
forward`.
            **kwargs: Keyword arguments, passed to the hypernet forward
                function.

        Returns:
            See docstring of method
            :meth:`hypnettorch.hnets.hnet_interface.HyperNetInterface.forward`.
        """
        if distilled_params is not None:
            raise NotImplementedError()
        if condition is not None:
            raise NotImplementedError()

        mnet_weights = None
        hnet_weights = None
        if weights is not None:
            if isinstance(weights, dict):
                raise NotImplementedError()
            assert len(weights) == len(self.param_shapes)

            nmw = len(self.internal_mnet.param_shapes)
            mnet_weights = weights[:nmw]
            hnet_weights = weights[nmw:]

        h = self.internal_mnet.forward(x, weights=mnet_weights)

        return self.internal_hnet.forward(h, weights=hnet_weights, **kwargs)

if __name__ == '__main__':
    pass



#!/usr/bin/env python3
# Copyright 2021 Hamza Keurti
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
# @title          :data/regression2d_heteroscedastic.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :31/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
2D Regression Dataset with Heteroscedastic Gaussian Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

from hypnettorch.data.dataset import Dataset
import numpy as np
from data.regression2d_gaussian import BivariateToyRegression


# BivariateToyRegression is the base class, to be derived from it:
# - BivariateBimodalRegression
# - BivariateHeteroscedasticRegression
# When inheriting, override init


class BivariateHeteroscedasticRegression(BivariateToyRegression):
    def __init__(self, train_inter=[-10, 10], num_train=20,
                 test_inter=[-10, 10], num_test=80, val_inter=None,
                 num_val=None,
                 map_function=lambda x: np.hstack([x, x]), cov=0, rseed=None,
                 perturb_test_val=True):
        """Generate a new regression dataset 
        from scalar x to depending bivariate y.
        Noise is a bivariate gaussian with covariance depending on x 
        according to cov_function.

        The input data x will be uniformly drawn for train samples and
        equidistant for test samples. The user has to specify a function that
        will map this random input data onto output samples y.

        Args:
            train_inter: A tuple, representing the interval from which x
                samples are drawn in the training set. Note, this range will
                apply to all input dimensions.
            num_train: Number of training samples.
            test_inter: A tuple, representing the interval from which x
                samples are drawn in the test set. Note, this range will
                apply to all input dimensions.
            num_test: Number of test samples.
            val_inter (optional): See parameter `test_inter`. If set, this
                argument leads to the construction of a validation set. Note,
                option `num_val` need to be specified as well.
            num_val (optional): Number of validation samples.
            map_function: A function handle that receives input
                samples and maps them to output samples.
            cov: If not zero, Gaussian white noise with this covariance will 
                be added to the training outputs. 
                If scalar, the covariance is the identity scaled by cov.
                If 2d, the covariance is diag(cov).
                If 2x2, cov is the covariance.
            cov_function: transforms x to a transformation matrix of the 
                noise covariance. 
                let :math:`R(x) = cov\_function(x)`. 
                If :math:`R(x) = diag(\lambda_1(x),\lambda_2(x))`,
                Then the 
                    

            rseed: If ``None``, the current random state of numpy is used to
                   generate the data. Otherwise, a new random state with the
                   given seed is generated.
        """
        super().__init__()

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)

        train_x = rand.uniform(low=train_inter[0], high=train_inter[1],
                               size=(num_train, 1))
        test_x = np.linspace(start=test_inter[0], stop=test_inter[1],
                             num=num_test).reshape((num_test, 1))

        train_y = map_function(train_x)
        test_y = map_function(test_x)

        # perturb training output
        if not hasattr(cov, '__len__'):  # if scalar
            cov = cov * np.eye(2)
        cov = np.array(cov)
        if cov.sum() > 0:
            if cov.shape == (2,):
                cov = np.diag(cov)
            assert(cov.shape == (2, 2))
            self._cov = cov


            train_eps = rand.multivariate_normal(mean=np.array([0,0]), cov=cov,
                                                    size=num_train)
            rots = self._rot_map(train_x)
            train_eps = np.einsum('ij,ikj->ik',train_eps,rots)
            train_y += train_eps

            if perturb_test_val:
                test_eps = rand.multivariate_normal(mean=np.array([0,0]), cov=cov,
                                        size=num_test)
                rots = self._rot_map(test_x)
                test_eps = np.einsum('ij,ikj->ik',test_eps,rots)
                test_y += test_eps

        # Create validation data if requested.
        if num_val is not None:
            val_x = np.linspace(start=val_inter[0], stop=val_inter[1],
                                num=num_val).reshape((num_val, 1))
            val_y = map_function(val_x)

            if perturb_test_val:
                val_eps = rand.multivariate_normal(mean=np.array([0,0]), cov=cov,
                                        size=num_val)
                rots = self._rot_map(val_x)
                val_eps = np.einsum('ij,ikj->ik',val_eps,rots)
                val_y += val_eps


            in_data = np.vstack([train_x, test_x, val_x])
            out_data = np.vstack([train_y, test_y, val_y])
        else:
            in_data = np.vstack([train_x, test_x])
            out_data = np.vstack([train_y, test_y])

        # Specify internal data structure.
        self._data['classification'] = False
        self._data['sequence'] = False
        self._data['in_data'] = in_data
        self._data['in_shape'] = [1]
        self._data['out_data'] = out_data
        self._data['out_shape'] = [2]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)

        self._map = map_function
        self._train_inter = train_inter
        self._test_inter = test_inter
        self._val_inter = val_inter

    def _get_rot_matrix(self,angle):
        """
        2D rotation matrix from angle in rad.

        With :math:`R` being the returned rotation matrix,
        A 2D line vector :math:`x` rotates through :math:`x*R^T`
        :param angle: Angle of 2D rotation.
        :return: A 2x2 rotation matrix. 
        """
        return np.array([
            [np.cos(angle),-np.sin(angle)],
            [np.sin(angle),np.cos(angle)]
            ])
    
    def _rot_map(self,x):
        """
        Maps array of x values to array of rotation matrices

        """
        return np.rollaxis(self._get_rot_matrix(x).squeeze(),2,0)

    def _get_cov_matrix(self,x):
        """
        get cov_matrix at position x.
        """
        rot = self._get_rot_matrix(x).squeeze()
        if len(rot.shape) > 2:
            rot = np.rollaxis(rot,2,0)
        return np.einsum('...j,...kj->...k',rot@self._cov,rot)
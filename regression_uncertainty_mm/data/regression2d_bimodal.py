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
# @title          :data/regression2d_bimodal.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :27/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
2D Regression Dataset with bimodal Gaussian Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
from data.regression2d_gaussian import BivariateToyRegression
import numpy as np

# BivariateToyRegression is the base class, to be derived from it:
# - BimodalBivariateRegression
# - BimodalHeteroscedasticRegression
# When inheriting, override init


class BivariateBimodalRegression(BivariateToyRegression):
    def __init__(self, train_inter=[-10, 10], num_train=20,
                 test_inter=[-10, 10], num_test=80, val_inter=None,
                 num_val=None, map_function=lambda x: np.hstack([x, x]), 
                 cov=np.array([5,0.5]), offset = 2, rot_angle = 0.78,
                 rseed=None, perturb_test_val=True):
        """Generate a new regression dataset 
        from scalar x to depending bivariate y.

        The input data x will be uniformly drawn for train samples and
        equidistant for test samples. The user has to specify a function that
        will map this random input data onto output samples y.

        Default parameters for the noise give a bimodal distribution that 
        looks marginally gaussian.
        
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
            offset: Noise will have two modes with means symmetrical to y=0 
                separated with means located at :math:`y = \pm offset` 
                prior to rotation by :param rot_angle:.
            rot_angle: Rotation angle by which to rotate the noise vector.
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
        self._rot_matrix = self._get_rot_matrix(rot_angle)
        self._mean1 = self._rot_matrix @ np.array([0,offset])
        self._mean2 = self._rot_matrix @ np.array([0,-offset])

        if not hasattr(cov, '__len__'):  # if scalar
            cov = cov * np.eye(2)
        cov = np.array(cov)
        if cov.sum() > 0:
            if cov.shape == (2,):
                cov = np.diag(cov)
            assert(cov.shape == (2, 2))
            self._cov = self._rot_matrix @ cov @ self._rot_matrix.T 

            eps1 = rand.multivariate_normal(mean=self._mean1, cov=self._cov,
                                                    size=num_train)
            eps2 = rand.multivariate_normal(mean=self._mean2, cov=self._cov,
                                                    size=num_train)
            b = rand.binomial(n=1,p=0.5,size = [num_train,1])
            
            train_eps = b*eps1 + (1-b)*eps2
            # train_eps = train_eps @ self._rot_matrix.T

            train_y += train_eps


            if perturb_test_val:
                eps1 = rand.multivariate_normal(mean=self._mean1, cov=self._cov,
                                    size=num_test)
                eps2 = rand.multivariate_normal(mean=self._mean2, cov=self._cov,
                                    size=num_test)
                b = rand.binomial(n=1,p=0.5,size = [num_test,1])

                test_eps = b*eps1 + (1-b)*eps2
                test_y += test_eps
        # Create validation data if requested.
        if num_val is not None:
            val_x = np.linspace(start=val_inter[0], stop=val_inter[1],
                                num=num_val).reshape((num_val, 1))
            val_y = map_function(val_x)
            
            if perturb_test_val and cov.sum()>0:
                eps1 = rand.multivariate_normal(mean=self._mean1, cov=self._cov,
                                    size=num_val)
                eps2 = rand.multivariate_normal(mean=self._mean2, cov=self._cov,
                                    size=num_val)
                b = rand.binomial(n=1,p=0.5,size = [num_val,1])

                val_eps = b*eps1 + (1-b)*eps2
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
        self._offset = offset
        self._rot_angle = rot_angle

    
    @property
    def offset(self):
        """
        Half the distance between the two modes means.
        """
        return self._offset
    
    @property
    def rot_angle(self):
        """
        Rotation angle of the noise distribution around its symmetry center.
        """
        return self._rot_angle
    
    @property
    def rot_matrix(self):
        """
        2D rotation matrix corresponding to the specified rotation angle 
        by which the noise distribution is rotated.
        """
        return self._rot_matrix
    
    @property
    def cov_matrix(self):
        """
        Covariance matrix of each mode. Covariance is the same for both modes.

        .. note::
            This is not the covariance input during initialization. 
            This is the covariance obtained after rotating the noise modes.
            It relates to the input covariance through:
            .. math::
                cov \leftarrow R\times cov \times R^T
            where :math:`R` is the rotation matrix obtained from the specified 
            rotation angle and can be obtained through :property:`rot_matrix`.
        """
        return self._cov
    
    @property
    def mean_mode1(self):
        """
        Mean of first mode. (2D position vector)
        """
        return self._mean1
    
    @property
    def mean_mode2(self):
        """
        Mean of second mode. (2D position vector)
        """
        return self._mean2

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
            


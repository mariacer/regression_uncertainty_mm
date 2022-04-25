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
# @title          :data/regression2d_gaussian.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :27/08/2021
# @version        :1.0
# @python_version :3.7.4
"""
2D Regression Dataset with Gaussian Error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""

from hypnettorch.data.dataset import Dataset
import numpy as np

# BivariateToyRegression is the base class, to be derived from it:
# - BimodalBivariateRegression
# - BimodalHeteroscedasticRegression
# When inheriting, override init


class BivariateToyRegression(Dataset):
    def __init__(self, train_inter=[-10, 10], num_train=20,
                 test_inter=[-10, 10], num_test=80, val_inter=None,
                 num_val=None,
                 map_function=lambda x: np.hstack([x, x]), cov=0, rseed=None,
                 perturb_test_val=True):
        """Generate a new regression dataset 
        from scalar x to depending bivariate y.

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
            
            train_eps = rand.multivariate_normal(mean=np.zeros(shape=2), 
                                    cov=self._cov, size=num_train)
            train_y += train_eps

            if perturb_test_val:
                test_eps = rand.multivariate_normal(mean=np.zeros(shape=2), 
                                        cov=self._cov, size=num_test)
                test_y += test_eps


        # Create validation data if requested.
        if num_val is not None:
            val_x = np.linspace(start=val_inter[0], stop=val_inter[1],
                                num=num_val).reshape((num_val, 1))
            val_y = map_function(val_x)
            if perturb_test_val and cov.sum()>0:
                val_eps = rand.multivariate_normal(mean=np.zeros(shape=2), 
                                        cov=self._cov, size=num_val)
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

    @property
    def train_x_range(self):
        """The input range for training samples."""
        return self._train_inter

    @property
    def test_x_range(self):
        """The input range for test samples."""
        return self._test_inter

    @property
    def val_x_range(self):
        """The input range for validation samples."""
        return self._val_inter

    @property
    def cov_matrix(self):
        """
        Covariance matrix.
        """
        return self._cov
    
    def get_identifier(self):
        """Returns name of dataset"""
        return '2DRegression'

    def _plot_sample(self, fig, inner_grid, num_inner_plots, ind, inputs,
                     outputs, predictions, **kwargs):
        """Not implemented"""
        # We overwrote the plot_samples method, so there is no need to ever call
        # this method (it's just here because the baseclass requires its
        # existence).
        raise NotImplementedError('TODO implement')

    def plot_dataset(self, show=False):
        """TODO""" 
        pass

    def _get_function_vals(self, num_samples=100, x_range=None):
        """Get real function values for equidistant x values in a range that
        covers the test and training data. These values can be used to plot the
        ground truth function.

        Args:
            num_samples: Number of samples to be produced.
            x_range: If a specific range should be used to gather function
                values.

        Returns:
            x, y: Two numpy arrays containing the corresponding x and y values.
        """
        if x_range is None:
            min_x = min(self._train_inter[0], self._test_inter[0])
            max_x = max(self._train_inter[1], self._test_inter[1])
            if self.num_val_samples > 0:
                min_x = min(min_x, self._val_inter[0])
                max_x = max(max_x, self._val_inter[1])
        else:
            min_x = x_range[0]
            max_x = x_range[1]

        slack_x = 0.05 * (max_x - min_x)

        sample_x = np.linspace(start=min_x-slack_x, stop=max_x+slack_x,
                               num=num_samples).reshape((num_samples, 1))
        sample_y = self._map(sample_x)

        return sample_x, sample_y


if __name__ == '__main__':
    pass

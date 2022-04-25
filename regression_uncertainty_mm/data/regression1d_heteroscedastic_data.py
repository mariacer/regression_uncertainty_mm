#!/usr/bin/env python3
# Copyright 2021 Rafael Daetwyler
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
# @title          :data/regression1d_heteroscedastic_data.py
# @author         :rd
# @contact        :rafael.daetwyler@uzh.ch
# @created        :02/09/2021
# @version        :1.0
# @python_version :3.7.4
"""
1D Regression Dataset with heteroscedastic Laplace error
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.special.regression1d_bimodal_data` contains a data handler
for a CL toy regression problem. The user can construct individual datasets with
this data handler and use each of these datasets to train a model in a continual
learning setting.
"""
import numpy as np
from hypnettorch.data.special.regression1d_data import ToyRegression


class HeteroscedasticToyRegression(ToyRegression):
    """An instance of this class shall represent a simple regression task, but
    with a heteroscedastic Laplace error distribution.

    Attributes:
        train_x_range: The input range for training samples.
        test_x_range: The input range for test samples.
        val_x_range: The input range for validation samples.
    """
    def __init__(self, train_inter=[-10, 10], num_train=20,
                 test_inter=[-10, 10], num_test=80, val_inter=None,
                 num_val=None, map_function=lambda x : x, scale=0, factor=5,
                 rseed=None, perturb_test_val=False):
        """Generate a new dataset.

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
            scale: If not zero, Laplace white noise with this scale will be
                added to the training outputs.
            factor: The std of the noise will be this many times larger at the
                right side of the data as on the left side.
            rseed: If ``None``, the current random state of numpy is used to
                   generate the data. Otherwise, a new random state with the
                   given seed is generated.
        """
        super().__init__()

        assert(val_inter is None and num_val is None or \
               val_inter is not None and num_val is not None)

        if rseed is None:
            rand = np.random
        else:
            rand = np.random.RandomState(rseed)

        self._map = map_function
        self._train_inter = train_inter
        self._test_inter = test_inter
        self._val_inter = val_inter
        self._scale = scale
        self._factor = factor

        train_x = rand.uniform(low=train_inter[0], high=train_inter[1],
                               size=(num_train, 1))
        test_x = np.linspace(start=test_inter[0], stop=test_inter[1],
                             num=num_test).reshape((num_test, 1))

        train_y = map_function(train_x)
        test_y = map_function(test_x)

        # Perturb training outputs.
        if scale > 0:
            loc = np.zeros((num_train,1))
            scale = self.f(train_x)
            #train_eps = rand.normal(loc=loc, scale=scale)
            train_eps = rand.laplace(loc=loc, scale=scale)
            train_y += train_eps

            if perturb_test_val:
                loc = np.zeros((num_test,1))
                scale = self.f(test_x)
                scale[scale < 0] = 0.0
                #test_eps = rand.normal(loc=loc, scale=scale)
                test_eps = rand.laplace(loc=loc, scale=scale)
                test_y += test_eps

        # Create validation data if requested.
        if num_val is not None:
            val_x = np.linspace(start=val_inter[0], stop=val_inter[1],
                                num=num_val).reshape((num_val, 1))
            val_y = map_function(val_x)

            if perturb_test_val:
                loc = np.zeros((num_val, 1))
                scale = self.f(val_x)
                scale[scale < 0] = 0.0
                #val_eps = rand.normal(loc=loc, scale=scale)
                val_eps = rand.laplace(loc=loc, scale=scale)
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
        self._data['out_shape'] = [1]
        self._data['train_inds'] = np.arange(num_train)
        self._data['test_inds'] = np.arange(num_train, num_train + num_test)

        if num_val is not None:
            n_start = num_train + num_test
            self._data['val_inds'] = np.arange(n_start, n_start + num_val)

    def f(self, x):
        t0, t1 = self._train_inter
        return self._scale * (1 - (self._factor - 1) * (t0 - np.array(x)) / (t1 - t0))

    def get_identifier(self):
        """Returns the name of the dataset."""
        return '1D Heteroscedastic Regression'

    @staticmethod
    def plot_datasets(data_handlers, inputs=None, predictions=None, labels=None,
                      fun_xranges=None, show=True, filename=None,
                      figsize=(10, 6), publication_style=False):
        """Plot several datasets of this class in one plot.

        Args:
            data_handlers: A list of ToyRegression objects.
            inputs (optional): A list of numpy arrays representing inputs for
                each dataset.
            predictions (optional): A list of numpy arrays containing the
                predicted output values for the given input values.
            labels (optional): A label for each dataset.
            fun_xranges (optional): List of x ranges in which the true
                underlying function per dataset should be sketched.
            show: Whether the plot should be shown.
            filename (optional): If provided, the figure will be stored under
                this filename.
            figsize: A tuple, determining the size of the figure in inches.
            publication_style: Whether the plots should be in publication style.
        """
        n = len(data_handlers)
        assert((inputs is None and predictions is None) or \
               (inputs is not None and predictions is not None))
        assert((inputs is None or len(inputs) == n) and \
               (predictions is None or len(predictions) == n) and \
               (labels is None or len(labels) == n))
        assert(fun_xranges is None or len(fun_xranges) == n)

        # Set-up matplotlib to adhere to our graphical conventions.
        #misc.configure_matplotlib_params(fig_size=1.2*np.array([1.6, 1]),
        #                                 font_size=8)

        # Get a colorscheme from colorbrewer2.org.
        colors = misc.get_colorbrewer2_colors(family='Dark2')
        if n > len(colors):
            warn('Changing to automatic color scheme as we don\'t have ' +
                 'as many manual colors as tasks.')
            colors = cm.rainbow(np.linspace(0, 1, n))

        if publication_style:
            ts, lw, ms = 60, 15, 140 # text fontsize, line width, marker size
            figsize = (12, 6)
        else:
            ts, lw, ms = 12, 2, 15

        fig, axes = plt.subplots(figsize=figsize)
        plt.title('1D regression', size=ts, pad=ts)

        phandlers = []
        plabels = []

        for i, data in enumerate(data_handlers):
            if labels is not None:
                lbl = labels[i]
            else:
                lbl = 'Function %d' % i

            fun_xrange = None
            if fun_xranges is not None:
                fun_xrange = fun_xranges[i]
            sample_x, sample_y = data._get_function_vals(x_range=fun_xrange)
            p, = plt.plot(sample_x, sample_y, color=colors[i],
                          linestyle='dashed', linewidth=lw/3)

            phandlers.append(p)
            plabels.append(lbl)
            if inputs is not None:
                p = plt.scatter(inputs[i], predictions[i], color=colors[i],
                    s=ms)
                pred_mu = predictions[i][:, 0]
                pred_sigma = np.exp(0.5 * predictions[i][:, 1])
                plt.fill_between(inputs[i][:,0], pred_mu - pred_sigma,
                                 pred_mu + pred_sigma, alpha=0.2, color=colors[i])
                phandlers.append(p)
                plabels.append('Predictions')

        if publication_style:
            axes.grid(False)
            axes.set_facecolor('w')
            axes.axhline(y=axes.get_ylim()[0], color='k', lw=lw)
            axes.axvline(x=axes.get_xlim()[0], color='k', lw=lw)
            if len(data_handlers)==3:
                plt.yticks([-1, 0, 1], fontsize=ts)
                plt.xticks([-2.5, 0, 2.5], fontsize=ts)
            else:
                for tick in axes.yaxis.get_major_ticks():
                    tick.label.set_fontsize(ts) 
                for tick in axes.xaxis.get_major_ticks():
                    tick.label.set_fontsize(ts) 
            axes.tick_params(axis='both', length=lw, direction='out', width=lw/2.)
        else:
            plt.legend(phandlers, plabels)

        plt.xlabel('$x$', fontsize=ts)
        plt.ylabel('$y$', fontsize=ts)
        plt.tight_layout()

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

        if show:
            plt.show()

if __name__ == '__main__':
    pass



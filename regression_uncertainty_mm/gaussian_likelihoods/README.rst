Regression experiments with Gaussian Likelihoods
************************************************

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

Performing regression experiments with Gaussian likelihood models. Two possibilities are considered:

* **Homoscedastic variance**: The model only generates the input-dependent mean, and the variance is constant across inputs and preset. It is given by the command line argument ``--pred_dist_std``. These experiments can be run using the script ``train_gl.py``.
* **Heteroscedastic variance**: The model generates both an input-dependent mean and an input-dependent variance.  These experiments can be run using the script ``train_ghl.py``.

Note that the models generated are Bayesian by default. However, a deterministic model can be used by setting the option ``--mean_only``.

Furthermore, regression can be done using one of two toy regression datasets based on a cubic polynomial. In the basic setting, a polynomial with a single mode is generated. However if desired a bimodal distribution can be used by setting the option ``--noise=bimodal``.

Unimodal 1D Toy Experiments
---------------------------

Please run the following command to see the available options for running 1D toy experiments.

.. code-block:: console

    $ python3 train_gl.py --help

Deterministic Gaussian Likelihood (homoscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

    $ python3 train_gl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=20001 --lr=0.0001 --mlp_arch=50,50 --net_act=relu --val_iter=1000 --num_train=20 --mean_only --noise=gaussian

Deterministic Gaussian Likelihood (heteroscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --batch_size=32 --n_iter=20001 --lr=0.0001 --mlp_arch=100,100 --net_act=sigmoid --val_iter=1000 --num_train=20 --mean_only --noise=gaussian
	
Gaussian Likelihood with BbB(homoscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console
	
	$ python3 train_gl.py --n_iter=20001 --lr=0.01 --train_sample_size=10 --mlp_arch=10,10 --net_act=relu --val_iter=1000 --val_sample_size=100 --num_train=20 --noise=gaussian

Gaussian Likelihood with BbB (heteroscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --batch_size=16 --n_iter=20001 --lr=0.001 --train_sample_size=1 --prior_variance=1.0 --kl_scale=0.01 --mlp_arch=100 --net_act=relu --val_iter=1000 --val_sample_size=100 --num_train=20 --noise=gaussian
	
Bimodal 1D Toy Experiments
--------------------------

Deterministic Gaussian Likelihood (homoscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **20 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=20001 --lr=0.01 --mlp_arch=20,20,20 --net_act=sigmoid --val_iter=1000 --num_train=20 --mean_only --noise=bimodal

The following run with **50 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --batch_size=32 --n_iter=20001 --lr=1e-05 --mlp_arch=100,100 --net_act=relu --val_iter=1000 --num_train=50 --mean_only --noise=bimodal

The following run with **1000 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=4001 --lr=0.01 --mlp_arch=20,20,20 --net_act=sigmoid --val_iter=1000 --num_train=1000 --mean_only --noise=bimodal

The following run with **1000 training points** and ground-truth predictive variance achieves lowest negative log-likelihood on the validation set for this model:

	python3 train_gl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=1001 --lr=0.01 --pred_dist_std=50.08991914547278 --mlp_arch=20,20,20 --net_act=sigmoid --val_iter=1000 --num_train=1000 --mean_only --noise=bimodal

Deterministic Gaussian Likelihood (heteroscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **20 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=10001 --lr=0.001 --mlp_arch=20,20,20 --net_act=relu --val_iter=1000 --num_train=20 --mean_only --noise=bimodal

The following run with **50 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=10001 --lr=0.001 --mlp_arch=50,50 --net_act=relu --val_iter=1000 --num_train=50 --mean_only --noise=bimodal --publication_style

The following run with **1000 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=5001 --lr=0.01 --mlp_arch=20,20,20 --net_act=relu --val_iter=1000 --num_train=1000 --mean_only --noise=bimodal

Gaussian Likelihood with BbB(homoscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **20 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl.py --n_iter=10001 --lr=0.01 --train_sample_size=1 --mlp_arch=50,50 --net_act=relu --val_iter=1000 --val_sample_size=100 --num_train=20 --noise=bimodal

The following run with **50 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl.py --disable_lrt_test --batch_size=32 --n_iter=10001 --lr=0.0001 --train_sample_size=1 --prior_variance=1.0 --local_reparam_trick --kl_scale=1.0 --mlp_arch=100 --net_act=relu --val_iter=1000 --val_sample_size=100 --num_train=50 --noise=bimodal --publication_style
	
The following run with **1000 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl.py --disable_lrt_test --n_iter=3001 --lr=0.01 --train_sample_size=20 --local_reparam_trick --mlp_arch=20,20,20 --net_act=relu --val_iter=1000 --val_sample_size=100 --num_train=1000 --noise=bimodal
	
Gaussian Likelihood with BbB (heteroscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **20 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --disable_lrt_test --n_iter=30001 --lr=0.01 --train_sample_size=10 --local_reparam_trick --mlp_arch=10,10 --net_act=sigmoid --val_iter=1000 --val_sample_size=100 --num_train=20 --noise=bimodal

The following run with **50 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --disable_lrt_test --batch_size=32 --n_iter=30001 --lr=0.001 --train_sample_size=10 --prior_variance=1.0 --local_reparam_trick --kl_scale=0.01 --mlp_arch=10,10 --net_act=relu --val_iter=1000 --val_sample_size=100 --num_train=50 --noise=bimodal
	
The following run with **1000 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl.py --disable_lrt_test --n_iter=30001 --lr=0.01 --train_sample_size=10 --local_reparam_trick --mlp_arch=10,10 --net_act=relu --val_iter=1000 --val_sample_size=100 --num_train=1000 --noise=bimodal
	
Bimodal 2D Toy Experiments
--------------------------

Please run the following command to see the available options for running 2D toy experiments.

.. code-block:: console

    $ python3 train_gl_2d.py --help

Deterministic Gaussian Likelihood (homoscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl_2d.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=10001 --lr=0.01 --pred_dist_std=17 --mlp_arch="20,20,20" --net_act=relu --val_iter=1000 --num_train=3000 --mean_only --noise=bimodal --offset=15 --cov=300,20

Deterministic Gaussian Likelihood (heteroscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl_2d.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=100001 --lr=0.0001 --mlp_arch=20,20,20 --net_act=relu --val_iter=1000 --num_train=3000 --mean_only --noise=bimodal --offset=15 --cov=300,20

Steering Angle Prediction Experiments
-------------------------------------

Please run the following command to see the available options for running 1D toy experiments.

.. code-block:: console

    $ python3 train_gl_udacity.py --help

Deterministic Gaussian Likelihood (homoscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl_udacity.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --batch_size=32 --epochs=20 --lr=0.0001 --adam_beta1=0.7 --clip_grad_norm=-1 --pred_dist_std=0.05 --net_type=iresnet --iresnet_use_fc_bias --store_models --use_empty_test_set --num_plotted_predictions=8 --mean_only


Steering Angle Prediction Experiments
-------------------------------------

Deterministic Gaussian Likelihood (homoscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-18** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_gl_udacity.py --use_empty_test_set --num_plotted_predictions=8 --mean_only --batch_size=64 --epochs=30 --lr=0.0001 --adam_beta1=0.5 --clip_grad_norm=100.0 --pred_dist_std=0.02 --kl_scale=0 --net_type=iresnet --iresnet_use_fc_bias --net_act=relu --train_sample_size=1 --val_sample_size=1

Deterministic Gaussian Likelihood (heteroscedastic noise)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with a **Resnet-18** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

	$ python3 train_ghl_udacity.py --use_empty_test_set --mean_only --val_set_size=5000 --batch_size=32 --epochs=30 --lr=0.0001 --adam_beta1=0.9 --clip_grad_value=-1 --clip_grad_norm=-1.0 --prior_variance=1.0 --pred_dist_std=3 --kl_scale=0 --net_type="iresnet" --iresnet_use_fc_bias --net_act="relu" --train_sample_size=1 --val_sample_size=1



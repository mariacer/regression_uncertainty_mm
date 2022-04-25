Normalizing Flows
*****************

.. Comment: Only the README content after the inclusion marker below will be added to the documentation by sphinx.
.. content-inclusion-marker-do-not-remove

Performing regression experiments with a flexible likelihood parametrized by a normalizing flow. The architecture of the system is the following:

.. image:: figures/nf.png
  :width: 400
  :alt: The architecture of our regression system based on normalizing flows.

Note that the models generated are Bayesian by default (i.e. the hypernetwork is Bayesian). However, a deterministic model can be used by setting the option ``--mean_only``.

Furthermore, regression can be done using one of two toy regression datasets based on a cubic polynomial. In the basic setting, a polynomial with a single mode is generated. However if desired a bimodal distribution can be used by setting the option ``--noise=bimodal``.

Unimodal 1D Toy Experiments
---------------------------

Please run the following command to see the available options for running 1D toy experiments.

.. code-block:: console

    $ python3 train.py --help

Deterministic hypernetwork and normalizing flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

    $ python3 train.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=7001 --lr=0.01 --clip_grad_norm=-1 --hmlp_arch=10,10,10 --hnet_dropout_rate=-1 --flow_depth=5 --flow_layer_type=splines --val_iter=1000 --num_train=20 --mean_only --noise=gaussian

Bayesian hypernetwork and normalizing flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

    $ python3 train.py --disable_lrt_test --n_iter=20001 --lr=0.01 --clip_grad_norm=-1 --train_sample_size=1 --local_reparam_trick --hmlp_arch=20,20 --hnet_dropout_rate=0.2 --flow_depth=1 --flow_layer_type=splines --val_iter=1000 --val_sample_size=100 --num_train=20 --noise=gaussian
  
Bimodal 1D Toy Experiments
--------------------------

Deterministic hypernetwork and normalizing flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **20 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

  $ python3 train.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=30001 --lr=0.01 --clip_grad_norm=1 --hmlp_arch=10,10 --hnet_dropout_rate=-1 --flow_depth=1 --flow_layer_type=splines --val_iter=1000 --num_train=20 --mean_only --noise=bimodal

The following run with **50 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

  $ python3 train.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --batch_size=32 --n_iter=10001 --lr=0.01 --clip_grad_norm=1 --hmlp_arch=100,100 --hnet_dropout_rate=-1 --hnet_net_act=sigmoid --flow_depth=3 --flow_layer_type=splines --val_iter=1000 --num_train=50 --mean_only --noise=bimodal --publication_style

The following run with **1000 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

  $ python3 train.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=50001 --lr=0.001 --clip_grad_norm=100 --hmlp_arch=10,10 --hnet_dropout_rate=-1 --flow_depth=3 --flow_layer_type=splines --val_iter=1000 --num_train=1000 --mean_only --noise=bimodal

Bayesian hypernetwork and normalizing flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **20 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

  $ python3 train.py --disable_lrt_test --n_iter=20001 --lr=0.0001 --clip_grad_norm=-1 --train_sample_size=10 --local_reparam_trick --hmlp_arch=10,10 --hnet_dropout_rate=-1 --flow_depth=1 --flow_layer_type=splines --val_iter=1000 --val_sample_size=190 --no_plots --num_train=20 --noise=bimodal

The following run with **50 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

  $ python3 train.py --disable_lrt_test --batch_size=32 --n_iter=10001 --lr=0.001 --clip_grad_norm=100 --train_sample_size=1 --prior_variance=1.0 --local_reparam_trick --kl_scale=0.01 --hmlp_arch=100,100 --hnet_dropout_rate=-1 --hnet_net_act=sigmoid --flow_depth=2 --flow_layer_type=splines --val_iter=1000 --val_sample_size=190 --num_train=50 --noise=bimodal

The following run with **1000 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

  $ python3 train.py --n_iter=50001 --lr=0.001 --clip_grad_norm=1 --train_sample_size=1 --hmlp_arch=5,5 --hnet_dropout_rate=-1 --flow_depth=3 --flow_layer_type=splines --val_iter=1000 --val_sample_size=190 --no_plots --num_train=1000 --noise=bimodal
  
Bimodal 2D Toy Experiments
--------------------------

Please run the following command to see the available options for running 2D toy experiments.

.. code-block:: console

    $ python3 train_2d.py --help

Deterministic hypernetwork and normalizing flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following run with **3000 training points** achieves lowest negative log-likelihood on the validation set for this model:

.. code-block:: console

  $ python3 train_2d.py --kl_scale=0 --train_sample_size=1 --val_sample_size=1 --n_iter=10001 --lr=0.01 --clip_grad_norm=100 --hmlp_arch=20,20,20 --hnet_dropout_rate=-1 --conditioner_arch=5,5 --flow_depth=10 --flow_layer_type=splines --val_iter=1000 --num_train=3000 --mean_only --noise=bimodal --offset=15 --cov=300,20

Steering Angle Prediction Experiments
-------------------------------------

Please run the following command to see the available options for running 1D toy experiments.

.. code-block:: console

    $ python3 train_udacity.py --help

Steering Angle Prediction Experiments
-------------------------------------

The following run with a **Resnet-18** achieves lowest negative log-likelihood on the validation set for this model:

    $ python3 train_udacity.py --mean_only --batch_size=16 --n_iter=1000 --epochs=30 --lr=0.0001 --adam_beta1=0.5 --clip_grad_value=-1 --clip_grad_norm=-1.0 --kl_scale=0 --flow_depth=15 --flow_layer_type="splines" --train_sample_size=1 --val_sample_size=1 --net_type="iresnet" --iresnet_use_fc_bias --net_act="relu" --no_bias --dropout_rate=-1 --hmlp_arch="" --hnet_net_act="relu" --hnet_dropout_rate=-1 --hmlp_uncond_in_size=10

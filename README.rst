===============================
VFWI
===============================

Variational Full-waveform inversion and travel time tomography

Authors
----------
 - Xin Zhang x.zhang2@ed.ac.uk

Requirements
------------
Cython, dask


Install
------------

In the ``VFWI`` folder, run


.. code-block:: sh
	
    sh setup.sh

This builds up the VFWI package, but does not install the package into the python path.
As a result, to use the package we need to tell python where the package is. For example,
when running scripts, do

.. code-block:: python
    
    PYTHONPATH=/your/vfwi/path python vfwi_example.py

See examples in ``tests`` folder. Instead you may want to install the package,

.. code-block:: sh

    sh setup.sh install

This will install the package into your python path, after which the package can be used directly
in your scripts.

Variational Inversion
---------------------
This package implements three different variational inference methods: ``ADVI (mean field)``,
``SVGD``, and ``Stochastic SVGD (sSVGD)``. To use them,

.. code-block:: python

    from vfwi.pyvi.svgd import SVGD, sSVGD
    from vfwi.pyvi.advi import ADVI

All methods require a function that takes model parameters as input and calculates the gradients of logarithm
posterior pdf function w.r.t parameters. For example,

.. code-block:: python
    
    def dlnprob(theta):
        # theta has a shape of (num_of_particles, num_of_parameters)
        # some calculation of theta
        return loss, grad, None

where loss is the misfit value (or negative logarithm of posterior pdf value), grad is the gradient. The third
output is used to return other auxiliary variables (e.g., a mask array), and can be safely ignored. Thereafter,

.. code-block:: python

    svgd = SVGD(dlnprob, kernel='rbf')

This creates a SVGD method which uses radial basis function kernel. To sample the posterior pdf,

.. code-block:: python

    losses = svgd.sample(x0, n_iter=1000, stepsize=0.01, optimizer='sgd')

where ``x0`` is a variable containing starting particles with a shape of ``(num_of_particles, num_of_parameters)``. This
will run the SVGD algorithm for 1,000 iterations using stochastic gradient descent (sgd) algorithm. Supported optimization
algorithms include ``sgd``, ``adagrad``, ``adadelta`` and ``adam``. To use sSVGD algorithm,

.. code-block:: python

    ssvgd = sSVGD(dlnprob, kernel='rbf')
    losses = ssvgd.sample(x0, n_iter=2000, stepsize=0.01, burn_in=1000)

This will sample the posterior using sSVGD method for 2,000 iterations with a burn-in period of 1,000.

Examples
---------
- For a complete 2D Full-waveform inversion example, please see the example in ``tests/fwi2d``. 
- For a complete 2D travel time tomography example, please see the example in ``tests/tomo2d``.
- For an example implementation of 3D Full-waveform inversion, please see the example in ``tests/fwi3d``. Note
  that this requires users to provide an external 3D FWI code to calculate misfit values and gradients. See details
  in ``vfwi/fwi``.

References
----------
- Zhang, X., & Curtis, A. (2020). Seismic tomography using variational inference methods. Journal of Geophysical Research: Solid Earth, 125(4), e2019JB018589.
- Zhang, X., Nawaz, M. A., Zhao, X., & Curtis, A. (2021). An introduction to variational inference in geophysical inverse problems. In Advances in Geophysics (Vol. 62, pp. 73-140). Elsevier.
- Zhang, X., & Curtis, A. (2020). Variational full-waveform inversion. Geophysical Journal International, 222(1), 406-411.

===============================
VIP
===============================

Variational Inversion Package, which implements variational methods for geophysical imaging problems, including
seismic travel time tomography and full waveform inversion.

Authors
----------
 - Xin Zhang x.zhang2@ed.ac.uk

Requirements
------------
Cython, Dask, H5py


Install
------------

In the ``VIP`` folder, run


.. code-block:: sh
	
    sh setup.sh

This builds up the VIP package, but does not install the package into your Python environment.
As a result, to use the package you need to tell python where the package is. For example,
when running scripts, do

.. code-block:: python
    
    PYTHONPATH=/your/VIP/path python vip_example.py

See examples in ``tests`` folder. Instead you may want to install the package,

.. code-block:: sh

    sh setup.sh install

This will install the package into your Python environment, after which the package can be used directly
in your scripts.

Variational Inversion
---------------------
This package implements three different variational inference methods: ``ADVI``,
``SVGD``, and ``stochastic SVGD (sSVGD)``. To use them,

.. code-block:: python

    from vip.pyvi.svgd import SVGD, sSVGD
    from vip.pyvi.advi import ADVI

All methods require a function that takes model parameters as input and calculates the gradients of logarithm
posterior pdf function w.r.t parameters. For example,

.. code-block:: python
    
    def dlnprob(theta):
        # theta has a shape of (num_of_particles, num_of_parameters)
        # some calculation of theta
        return logp, grad, None

where logp is the logarithm of (unnormalized) posterior pdf value (or negative misfit value), grad is the gradient. The third
output is used to return other auxiliary variables (e.g., a mask array), and can be safely ignored. Thereafter,

.. code-block:: python

    svgd = SVGD(dlnprob, kernel='rbf')

This creates a SVGD method which uses radial basis function kernel. To sample the posterior pdf,

.. code-block:: python

    losses, x = svgd.sample(x0, n_iter=1000, stepsize=0.01, optimizer='sgd')

where ``x0`` and ``x`` are variables containing starting and final particles with a shape of ``(num_of_particles, num_of_parameters)`` 
respectively. This will run the SVGD algorithm for 1,000 iterations using stochastic gradient descent (sgd) algorithm. Supported optimization
algorithms include ``sgd``, ``adagrad``, ``adadelta`` and ``adam``. To use sSVGD algorithm,

.. code-block:: python

    ssvgd = sSVGD(dlnprob, kernel='rbf')
    losses, x = ssvgd.sample(x0, n_iter=2000, stepsize=0.01, burn_in=1000)

This will sample the posterior using sSVGD method for 2,000 iterations with a burn-in period of 1,000. To use ADVI,

.. code-block:: python

    advi = ADVI(dlnprob, kernel='meanfield')
    losses, phi = advi.sample(n_iter=2000, stepsize=0.01, optimizer='adam')

This runs mean-field ADVI for 2,000 iterations using the ``adam`` optimization algorithm. The vector ``phi`` contains the mean (first half) 
and the logarithm of the standard deviation (second half) of the final Gaussian distribution. To use fullrank ADVI, set kernel to "fullrank".
In this case, assume the number of parameters is n, the first n elements of ``phi`` are the mean, and the rest n^2 elements are the Cholesky
decomposition (L) of the covariance matrix.

Examples
---------
- For a complete 2D Full-waveform inversion example, please see the example in ``tests/fwi2d``. 
- For a complete 2D travel time tomography example, please see the example in ``tests/tomo2d``.
- For an example implementation of 3D Full-waveform inversion, please see the example in ``tests/fwi3d``. Note
  that this requires users to provide an external 3D FWI code to calculate misfit values and gradients. See details
  in ``forward/fwi3d``.

References
----------
- Zhang, X., & Curtis, A. (2024). VIP-Variational Inversion Package with example implementations of Bayesian tomographic imaging. Seismica, 3(1).
- Zhang, X., & Curtis, A. (2020). Seismic tomography using variational inference methods. Journal of Geophysical Research: Solid Earth, 125(4), e2019JB018589.
- Zhang, X., Nawaz, M. A., Zhao, X., & Curtis, A. (2021). An introduction to variational inference in geophysical inverse problems. In Advances in Geophysics (Vol. 62, pp. 73-140). Elsevier.
- Zhang, X., Lomas, A., Zhou, M., Zheng, Y., & Curtis, A. (2023). 3-D Bayesian variational full waveform inversion. Geophysical Journal International, 234(1), 546-561.

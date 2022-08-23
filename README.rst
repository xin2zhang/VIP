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

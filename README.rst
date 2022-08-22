===============================
VFWI
===============================

variational Full-waveform inversion using Stein variational gradient descent

Authors
----------
 - Xin Zhang x.zhang2@ed.ac.uk

Requirements
------------
Cython, dask


Install
------------

In the svgd folder, build svgd related Cython code first


.. code-block:: sh
	
   	cd svgd
	python setup.py build_ext -i

There are 2D and 3D code in the respective folder. See details in the README file in each folder.
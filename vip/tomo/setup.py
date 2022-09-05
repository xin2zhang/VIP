from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from os import system
npy_include_dir = numpy.get_include()

# compile fortran code
make_comp = 'make'
print(make_comp)
system(make_comp)

ext_modules = [Extension("pyfm2d", ["pyfm2d.pyx"],
                         include_dirs = [npy_include_dir],
                         libraries=["gfortran","gomp"],
                         extra_objects=["fm2d_globalp.o", "fm2d_ttime.o", "fm2dray_cartesian.o","fm2d_wrapper.o"])]

setup(name = 'fast marching',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)

system('make clean')
system('rm -rf build')

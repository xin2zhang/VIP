from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
from os import system
npy_include_dir = numpy.get_include()

ext_modules = [Extension("aco2d", ["pyacofwi2D.pyx", "fdtd2D_modelling.c"],
                         include_dirs = [npy_include_dir],
                         language='c',
                         extra_compile_args=["-fopenmp"],
                         extra_link_args=["-fopenmp"],
                         #libraries=["gcc"],
                         #extra_objects=["fdtd2D_modelling.o"]
                         )]

setup(name = 'acousticfwi2D',
      cmdclass = {'build_ext': build_ext},
      ext_modules = ext_modules)

system('rm -rf build')

from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
from os import system

# compile fortran code
make_comp = 'make'
print(make_comp)
system('make clean')
system(make_comp)

extensions = [Extension("pykernel",
                        ["pykernel.pyx"],
                        #["pytrans.pyx","pykernel.pyx"],
                        include_dirs=[np.get_include()],
                        libraries=["gfortran","gomp"],
                        extra_compile_args=['-O3'],
                        extra_link_args=['-L/usr/lib64','-lm'],
                        extra_objects=["utils.o","transform.o","kernel.o"]),
            Extension("pytrans",
                        ["pytrans.pyx"],
                        #["pytrans.pyx","pykernel.pyx"],
                        include_dirs=[np.get_include()],
                        libraries=["gfortran","gomp"],
                        extra_compile_args=['-O3'],
                        extra_link_args=['-L/usr/lib64','-lm'],
                        extra_objects=["utils.o","transform.o","kernel.o"])
              ]

for e in extensions:
    e.cython_directives={'language_level': "3"}

setup(
    name = 'pysvgd',
    cmdclass = {'build_ext':build_ext},
    #ext_modules = cythonize(extensions, language_level = "3")
    ext_modules = extensions,
)

system('make clean')
system('rm -rf build')

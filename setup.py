# https://www.youtube.com/watch?v=mXuEoqK4bEc
# https://cython.readthedocs.io/en/latest/src/quickstart/build.html
# https://numpy.org/doc/stable/user/c-info.python-as-glue.html#cython
from Cython.Distutils import build_ext
from distutils.extension import Extension
from distutils.core import setup
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True
import numpy

setup(name='mine', description='Nothing',
      ext_modules=[Extension('ex7_solver_gauss_seidel_method_cython', ['ex7_solver_gauss_seidel_method_cython.pyx'],
                             include_dirs=[numpy.get_include()])],
      cmdclass = {'build_ext':build_ext})
# https://www.youtube.com/watch?v=mXuEoqK4bEc
# https://cython.readthedocs.io/en/latest/src/quickstart/build.html
# https://numpy.org/doc/stable/user/c-info.python-as-glue.html#cython

# python setup.py build_ext --inplace

from Cython.Distutils import build_ext
from distutils.extension import Extension
from distutils.core import setup
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True       # uncomment voor html file
import numpy

setup(name='mine', description='Nothing',
      ext_modules=[
            
                  Extension('ex6_solver_jacobi_method_cython', ['ex6_solver_jacobi_method_cython.pyx'],
                  include_dirs=[numpy.get_include()]
                  ),
                  Extension('ex7_solver_gauss_seidel_method_cython', ['ex7_solver_gauss_seidel_method_cython.pyx'],
                  include_dirs=[numpy.get_include()]
                  ),
                  Extension('ex8_solver_backward_gauss_seidel_method_cython', ['ex8_solver_backward_gauss_seidel_method_cython.pyx'],
                  include_dirs=[numpy.get_include()]
                  ),
                  Extension('ex9_solver_symmetric_gauss_seidel_method_cython', ['ex9_solver_symmetric_gauss_seidel_method_cython.pyx'],
                  include_dirs=[numpy.get_include()]
                  ),
                  Extension('ex12_solver_GMRES_cython', ['ex12_solver_GMRES_cython.pyx'],
                  include_dirs=[numpy.get_include()]
                  )],
      cmdclass = {'build_ext':build_ext, 'language_level':3})
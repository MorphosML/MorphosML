import os
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11

class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""
    def __str__(self):
        return pybind11.get_include()

ext_modules = [
    Extension(
        'morphosml._core',
        sources=[
            'cpp/src/vector.cpp',
            'cpp/src/matrix.cpp',
            'cpp/src/knn.cpp',
            'cpp/src/bindings.cpp',
        ],
        include_dirs=[
            get_pybind_include(),
            pybind11.get_include(),
            'cpp/include',
        ],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3', '-fPIC'],
        extra_link_args=['-std=c++17'],
    ),
]

def has_flag(compiler, flagname):
    """Return boolean if compiler supports a flag"""
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
        return True
    except Exception:
        return False
    finally:
        os.remove(fname)

class BuildExt(build_ext):
    """Custom build extension to add platform-specific flags"""
    def build_extensions(self):
        if sys.platform == 'darwin':
            for ext in self.extensions:
                ext.extra_compile_args.append('-mmacosx-version-min=10.9')
                ext.extra_link_args.append('-mmacosx-version-min=10.9')
        elif sys.platform == 'win32':
            for ext in self.extensions:
                ext.extra_compile_args.append('/std:c++17')
                ext.extra_compile_args.append('/O2')
        else:  # Linux
            pass
        super().build_extensions()

# Read README
try:
    with open('README.md', 'r', encoding='utf-8') as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "High-performance machine learning library with C++ core"

setup(
    name='morphosml',
    version='0.2.0',
    author='Gabriel Carmona',
    author_email='gabrielcarmonabittencourtpy@gmail.com',
    description='High-performance machine learning library with C++ core',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=['morphosml'],
    package_dir={'': 'src'},
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExt},
    install_requires=['numpy>=1.24.0'],
    zip_safe=False,
)
import os
import platform
import re
import sys

import numpy
from setuptools import setup, find_packages, Extension

print """
*******************************************************************
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

(c) 2016 Oxford Nanopore Technologies Ltd.
*******************************************************************
"""

# Get the version number from __init__.py
pkg_name = 'nanonet'
pkg_path = os.path.join(os.path.dirname(__file__), pkg_name)
verstrline = open(os.path.join(pkg_path, '__init__.py'), 'r').read()
vsre = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(vsre, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in "{}/__init__.py".'.format(pkg_name))

system = platform.system()
print "System is {}".format(system)
print "By default the 2D basecaller (standard and OpenCL) are not built."
print "To enable these use 'with2d' and 'opencl2d' command line options."
print

with_2d = True if 'with2d' in sys.argv else False
if with_2d:
    sys.argv.remove('with2d')

opencl_2d = True if 'opencl2d' in sys.argv else False
if opencl_2d:
    with_2d = True
    sys.argv.remove('opencl2d')

mingw = True if "mingw" in sys.argv else False
if mingw:
    sys.argv.remove('mingw')
    # patch distutils to force our compiler class. With distutils build
    #   command we can use the commandline option to set compiler but
    #   develop command does accept this. C++ extensions also aren't
    #   recognised as being unchanged meaning they get built twice with
    #       build --compiler=mingw32 develop
    #   all very annoying.
    import distutils.cygwinccompiler
    from nanoccompiler import Mingw64CCompiler

    distutils.cygwinccompiler.Mingw32CCompiler = Mingw64CCompiler
    distutils.ccompiler.get_default_compiler = lambda x: 'mingw32'

main_include = os.path.join(os.path.dirname(__file__), 'nanonet', 'include')
include_dirs = [main_include]
event_detect_include = []
boost_inc = []
boost_lib_path = []
boost_libs = []
opencl_include = []
opencl_lib_path = []
opencl_libs = []

c_compile_args = ['-pedantic', '-Wall', '-std=c99']
cpp_compile_args = []
optimisation = ['-DNDEBUG']

if system == 'Darwin':
    print "Adding OSX compile/link options"
    optimisation.extend(['-O3', '-fstrict-aliasing'])
    cpp_compile_args.extend(['-std=c++0x', '-Wno-unused-local-typedefs'])
    # may wish to edit - required for 2D
    boost_inc = ['/opt/local/include/']
    boost_libs.append('boost_python-mt')
elif system == 'Windows':
    event_detect_include.append(os.path.join(pkg_path, 'eventdetection'))
    if not mingw:
        print "Adding windows (MSVC) compile/link options"
        optimisation = ['/O2', '/Gs-']
        c_compile_args = ['/wd4820']
        cpp_compile_args.extend(['/EHsc', '/wd4996'])
        include_dirs.append(os.path.join(main_include, 'extras'))
        boost_location = os.path.join('c:', os.sep, 'local', 'boost_1_55_0')
        boost_lib_name = 'lib64-msvc-9.0'
        if opencl_2d:
            raise NotImplementedError('OpenCL 2D caller not currently supported on Windows with MSVC.')
    else:
        print "Adding windows (mingw64) compile/link options"
        optimisation.extend(['-O3', '-fstrict-aliasing'])
        c_compile_args.extend(['-DMS_WIN64', '-D_hypot=hypot'])
        cpp_compile_args.extend(['-DMS_WIN64', '-D_hypot=hypot', '-Wno-unused-local-typedefs'])
        boost_location = os.environ.get(
            'BOOST_ROOT', os.path.join('c:', os.sep, 'local', 'boost_1_55_0'))
        boost_lib_name = os.environ.get(
            'BOOST_LIB', os.path.join('stage', 'lib'))
        boost_libs.append(
            os.environ.get('BOOST_PYTHON', 'boost_python-mgw48-mt-1_55'))
        # may wish to edit - required for OpenCL 2D, this will compile
        #   but likely die at runtime.
        if opencl_2d:
            raise NotImplementedError('OpenCL 2D caller not currently supported on Windows with mingw64.')
        # nvidia_opencl = os.path.join('c:', os.sep,
        #    'Program Files', 'NVIDIA GPU Computing Toolkit', 'CUDA', 'v7.5')
        # opencl_include = [os.environ.get('OPENCL_INC', os.path.join(nvidia_opencl, 'include'))]
        # opencl_lib_path = [os.environ.get('OPENCL_LIB', os.path.join(nvidia_opencl, 'lib', 'x64'))]
        # opencl_libs.append('OpenCL')
    boost_lib_path = [os.path.join(boost_location, boost_lib_name)]
    boost_inc = [boost_location]
else:
    print "Adding Linux(?) compile/link options"
    optimisation.extend(['-O3', '-fstrict-aliasing'])
    cpp_compile_args.extend(['-std=c++0x', '-Wno-unused-local-typedefs'])
    boost_libs.append('boost_python')
    # may wish to edit - required for OpenCL 2D
    opencl_include = [os.environ.get('OPENCL_INC')]
    opencl_lib_path = [os.environ.get('OPENCL_LIB', os.path.join(os.sep, 'opt', 'intel', 'opencl'))]
    opencl_libs.append('OpenCL')
c_compile_args.extend(optimisation)
cpp_compile_args.extend(optimisation)

extensions = []

extensions.append(Extension(
    'nanonetfilters',
    sources=[os.path.join(pkg_path, 'eventdetection', 'filters.c')],
    include_dirs=include_dirs + event_detect_include,
    extra_compile_args=c_compile_args
))

extensions.append(Extension(
    'nanonetdecode',
    sources=[os.path.join(pkg_path, 'decoding.cpp')],
    include_dirs=include_dirs,
    extra_compile_args=cpp_compile_args
))

if with_2d:
    caller_2d_path = os.path.join('nanonet', 'caller_2d')
    extensions.append(Extension(
        'nanonet.caller_2d.viterbi_2d.viterbi_2d',
        include_dirs=[os.path.join(caller_2d_path, x) for x in
                      ('viterbi_2d', 'common')] +
                     [numpy.get_include()] + boost_inc + include_dirs,
        sources=[os.path.join(caller_2d_path, 'viterbi_2d', x) for x in
                 ('viterbi_2d_py.cpp', 'viterbi_2d.cpp')],
        depends=[os.path.join(caller_2d_path, x) for x in
                 ('viterbi_2d_py.h', 'viterbi_2d.h')] +
                [os.path.join(caller_2d_path, 'common', x) for x in
                 ('bp_tools.h', 'data_view.h', 'utils.h', 'view_numpy_arrays.h')],
        extra_compile_args=cpp_compile_args,
        library_dirs=boost_lib_path,
        libraries=boost_libs
    ))

    extensions.append(Extension(
        'nanonet.caller_2d.pair_align.pair_align',
        include_dirs=[os.path.join(caller_2d_path, 'pair_align')] +
                     boost_inc + include_dirs,
        sources=[os.path.join(caller_2d_path, 'pair_align', x) for x in
                 ('pair_align_py.cpp', 'nw_align.cpp', 'mm_align.cpp')],
        depends=[os.path.join(caller_2d_path, 'pair_align', x) for x in
                 ('pair_align_py.h', 'pair_align.h', 'nw_align.h', 'mm_align.h')],
        extra_compile_args=cpp_compile_args,
        library_dirs=boost_lib_path,
        libraries=boost_libs
    ))

    extensions.append(Extension(
        'nanonet.caller_2d.common.stub',
        include_dirs=[os.path.join(caller_2d_path, 'common')] +
                     [numpy.get_include()] + boost_inc + include_dirs,
        sources=[os.path.join(caller_2d_path, 'common', 'stub_py.cpp')],
        depends=[os.path.join(caller_2d_path, 'common', x) for x in
                 ('bp_tools.h', 'data_view.h', 'utils.h', 'view_numpy_arrays.h')],
        extra_compile_args=cpp_compile_args,
        library_dirs=boost_lib_path,
        libraries=boost_libs
    ))

if opencl_2d:
    print "Setting up OpenCL 2D basecall extension, this may need some tinkering"
    extensions.append(Extension(
        'nanonet.caller_2d.viterbi_2d_ocl.viterbi_2d_ocl',
        include_dirs=[os.path.join(caller_2d_path, x) for x in
                      ('viterbi_2d_ocl', 'common')] +
                     [numpy.get_include()] + boost_inc + include_dirs + opencl_include,
        sources=[os.path.join(caller_2d_path, 'viterbi_2d_ocl', x) for x in
                 ('viterbi_2d_ocl_py.cpp', 'viterbi_2d_ocl.cpp', 'proxyCL.cpp')],
        depends=[os.path.join(caller_2d_path, 'viterbi_2d_ocl', x) for x in
                 ('viterbi_2d_ocl.py.h', 'viterbi_2d_ocl.h', 'proxyCL.h')] +
                [os.path.join(caller_2d_path, 'common', x) for x in
                 ('bp_tools.h', 'data_view.h', 'utils.h', 'view_numpy_arrays.h')],
        extra_compile_args=cpp_compile_args,
        library_dirs=boost_lib_path + opencl_lib_path,
        libraries=boost_libs + opencl_libs
    ))

requires = [
    'h5py',
    'myriad >=0.1.2',
    'numpy',
]
extra_requires = {
    'currennt': ['netCDF4'],
    'watcher': ['watchdog'],
    'opencl': ['pyopencl'],
    'simulate': ['biopython'],
}

# Making a whl for windows
bdist_args = dict()
if system == 'Windows' and "bdist_wheel" in sys.argv:
    from setuptools import Distribution
    from distutils.spawn import find_executable
    from glob import glob


    class BinaryDistribution(Distribution):
        def is_pure(self):
            return False

        def has_ext_modules(self):
            return True


    blibs = [os.path.join(boost_location, boost_lib_name, 'lib{}.dll'.format(x)) for x in boost_libs]
    mingwdir = os.path.dirname(find_executable('gcc'))
    mingwlibs = glob(os.path.join(mingwdir, '*.dll'))
    mingwlibs = [os.path.join(mingwdir, x) for x in mingwlibs]
    dlls = [os.path.relpath(x) for x in blibs + mingwlibs]
    bdist_args = {
        'scripts': dlls,
        'distclass': BinaryDistribution
    }

setup(
    name='nanonet',
    version=version,
    description='A simple recurrent neural network based basecaller nanopore data.',
    maintainer='Chris Wright',
    maintainer_email='chris.wright@nanoporetech.com',
    url='http://www.nanoporetech.com',
    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
    package_data={'nanonet.data': ['nanonet/data/*']},
    include_package_data=True,
    tests_require=requires,
    install_requires=requires,
    extras_require=extra_requires,
    dependency_links=[],
    zip_safe=True,
    ext_modules=extensions,
    test_suite='discover_tests',
    entry_points={
        'console_scripts': [
            'nanonetcall = nanonet.nanonetcall:main',
            'nanonet2d = nanonet.nanonetcall_2d:main',
            'nanonettrain = nanonet.nanonettrain:main',
            'simulate_minion = nanonet.simulate.simulate_minion:main',
        ]
    },
    **bdist_args
)


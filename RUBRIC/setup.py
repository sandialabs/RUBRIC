import os

from setuptools import setup

setup(
    name='RUBRIC',
    version='1.0',
    packages=['RUBRIC'],
    url='https://github.com/ragak/RUBRIC',
    license='MPL 2.0',
    author='Harrison Edwards, Raga Krishnakumar, and Michael Bartsch',
    author_email='harrison.edwards@mail.utoronto.ca',
    description='Read-Until with Basecalling and Reference-Informed Criteria (RUBRIC)',
    install_requires=['configargparse', 'h5py', 'numpy', 'termcolor', 'thrift==0.9.2', 'ws4py', 'biopython', 'psutil']
)

# some extra lines to run the nanonet setup file...
cwd = os.getcwd()
cmd_string = os.path.join(cwd, 'RUBRIC', 'setup_nanonet.py')
cmd_string = 'python ' + cmd_string + ' develop'
os.system(cmd_string)




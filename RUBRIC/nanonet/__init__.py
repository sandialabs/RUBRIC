import os
import sys
import re
import subprocess

__version__ = '2.0.0'
__version_info__ = tuple([int(num) for num in __version__.split('.')])

try:
    import pyopencl as cl
except ImportError:
    cl = None

try:
    __currennt_exe__ = os.path.abspath(os.environ['CURRENNT'])
except KeyError:
    __currennt_exe__ = 'currennt'

def check_currennt():
    # Check we can run currennt
    try:
        with open(os.devnull, 'w') as devnull:
            subprocess.call([__currennt_exe__, '-h'], stdout=devnull, stderr=devnull)
    except OSError:
        raise OSError("Cannot execute currennt, it must be in your path as 'currennt' or set via the environment variable 'CURRENNT'.")


def run_currennt(currennt_cfg, device=0):
    sys.stdout.flush()
    os.environ["CURRENNT_CUDA_DEVICE"]="{}".format(device)
    cmd = [__currennt_exe__, currennt_cfg]
    with open(os.devnull, 'wb') as devnull:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=devnull)
        stdout, _ = p.communicate()
        p.wait()
        if p.returncode != 0:
            # On windows currennt fails to remove the cache file. Check for
            #   this and move on, else raise an error.
            e = subprocess.CalledProcessError(2, ' '.join(cmd))
            if os.name != 'nt':
                sys.stderr.write(stdout)
                raise e
            else:
                cache_file = re.match(
                    '(FAILED: boost::filesystem::remove.*: )"(.*)"',
                    stdout.splitlines()[-1])
                if cache_file is not None:
                    cache_file = cache_file.group(2)
                    sys.stderr.write('currennt failed to clear its cache, cleaning up {}\n'.format(cache_file))
                    os.unlink(cache_file)
                else:
                    sys.stderr.write(stdout)
                    raise e

def run_currennt_noisy(currennt_cfg, device=0):
    sys.stdout.flush()
    os.environ["CURRENNT_CUDA_DEVICE"]="{}".format(device)
    cmd = [__currennt_exe__, currennt_cfg]
    subprocess.check_call(cmd)

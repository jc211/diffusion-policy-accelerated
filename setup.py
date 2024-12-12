import sys
from setuptools import setup

BRIGHT_ORANGE = '\033[93m'
RESET_COLOR = '\033[0m'

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
except ImportError as e:
    error_message = "Error: Torch must be installed before installing this package.\n"
    colored_error_message = f"{BRIGHT_ORANGE}{error_message}{RESET_COLOR}"
    sys.stderr.write(colored_error_message)
    sys.exit(1)

setup(
    ext_modules=[
        CUDAExtension(
            name='diffusion_policy_accelerated.conv1d_gnm',
            sources=['csrc/conv1d_gnm.cpp', 'csrc/conv1d_gnm_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
)

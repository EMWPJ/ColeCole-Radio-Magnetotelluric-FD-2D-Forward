"""
Configuration parameters for RMT 2D FD Forward Modeling
射频大地电磁法二维正演配置参数
"""

import numpy as np

PI = np.pi

MU = PI * 4e-7

EPS0 = 8.854e-12

NX = 256
NZ = 256
PAD = 20
CORE_COLS = NX - 2 * PAD

COLE_COLE_DEFAULT = {
    'sigma0': 0.001,
    'm': 0.1,
    'tau': 1e-3,
    'c': 0.5,
    'epsilon_r': 5.0
}

X_RANGE = (-5000, 5000)
Z_DEPTH = 10000

FREQUENCIES_DEFAULT = np.logspace(4, 5.4, 20)

OUTPUT_DIR = './results'

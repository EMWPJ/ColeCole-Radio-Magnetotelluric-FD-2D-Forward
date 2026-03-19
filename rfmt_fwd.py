"""
RMT 2D FD Forward Modeling Entry Point
射频大地电磁法二维正演入口
"""

import numpy as np
from forward.modeling import rfmt_forward, save_results


def main():
    """主函数示例"""
    nx, nz = 256, 256
    pad = 20

    sigma = np.full((nz, nx), 0.001)

    terrain = np.full(nx, 26)

    frequencies = np.logspace(4, 5.4, 5)

    dy = np.ones(nx) * 50.0
    dz = np.ones(nz) * 50.0

    results = rfmt_forward(sigma, terrain, frequencies, dy, dz)

    save_results(results, 'result_core.txt')

    print(f"Results saved. Shape: {results.shape}")


if __name__ == '__main__':
    main()

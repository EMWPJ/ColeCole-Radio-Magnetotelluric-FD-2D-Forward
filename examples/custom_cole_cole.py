"""
Example: Using custom Cole-Cole parameters
使用自定义Cole-Cole参数
"""

import numpy as np
from forward.modeling import rfmt_forward, save_results


def run_with_cole_cole():
    """
    使用Cole-Cole参数运行正演
    """
    nx, nz = 256, 256

    sigma = np.full((nz, nx), 0.001)

    terrain = np.full(nx, 26)

    frequencies = np.logspace(4, 5.4, 10)

    dy = np.ones(nx) * 50.0
    dz = np.ones(nz) * 50.0

    sigma0 = 0.001
    m = 0.2
    tau = 1e-4
    c = 0.3
    epsilon_r = 5.0

    print(f"Cole-Cole parameters:")
    print(f"  sigma0 = {sigma0} S/m")
    print(f"  m = {m}")
    print(f"  tau = {tau} s")
    print(f"  c = {c}")
    print(f"  epsilon_r = {epsilon_r}")

    results = rfmt_forward(
        sigma, terrain, frequencies, dy, dz,
        sigma0=sigma0, m=m, tau=tau, c=c, epsilon_r=epsilon_r
    )

    save_results(results, 'cole_cole_result.txt')
    print(f"Results saved. Shape: {results.shape}")


if __name__ == '__main__':
    run_with_cole_cole()

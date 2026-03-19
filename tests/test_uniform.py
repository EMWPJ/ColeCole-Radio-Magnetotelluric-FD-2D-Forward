"""
Test: Uniform Half-space Analytical Solution
均匀半空间解析解验证
"""

import numpy as np
from forward.modeling import rfmt_forward, save_results


def analytical_resistivity(freq: float, rho: float = 1000.0,
                          mu: float = np.pi * 4e-7) -> float:
    """
    均匀半空间视电阻率解析解

    对于均匀半空间，视电阻率等于真电阻率

    Args:
        freq: 频率 (Hz)
        rho: 真电阻率 (Ω·m)
        mu: 磁导率 (H/m)

    Returns:
        视电阻率 (Ω·m)
    """
    return rho


def test_uniform_halfspace():
    """
    测试均匀半空间模型
    """
    nx, nz = 256, 256
    pad = 20

    rho0 = 1000.0
    sigma0 = 1.0 / rho0

    sigma = np.full((nz, nx), sigma0)

    terrain = np.full(nx, 26)

    frequencies = np.array([10000.0, 50000.0, 100000.0])

    dy = np.ones(nx) * 50.0
    dz = np.ones(nz) * 50.0

    print("Running uniform half-space forward...")
    results = rfmt_forward(sigma, terrain, frequencies, dy, dz)

    print(f"Results shape: {results.shape}")

    for i in range(len(frequencies)):
        fre = results[i, 0]
        rhoxy = results[i, 3]
        print(f"f={fre:.0f}Hz, rhoxy={rhoxy:.2f} Ω·m "
              f"(expected={rho0:.2f} Ω·m, "
              f"error={abs(rhoxy-rho0)/rho0*100:.2f}%)")

    save_results(results, 'test_uniform_result.txt')
    print("Results saved to test_uniform_result.txt")


if __name__ == '__main__':
    test_uniform_halfspace()

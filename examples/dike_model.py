"""
Example: Dike Model (论文1模型验证)
Dike模型 - 用于验证程序正确性
"""

import numpy as np
from forward.modeling import rfmt_forward, save_results


def build_dike_model():
    """
    构建Dike模型

    参考论文1: 背景电阻率10000 Ω·m，中间断层电阻率1000 Ω·m

    Returns:
        sigma: 电导率数组
        terrain: 地形数组
    """
    nx, nz = 256, 256
    pad = 20

    rho_background = 10000.0
    rho_dike = 1000.0
    sigma_background = 1.0 / rho_background
    sigma_dike = 1.0 / rho_dike

    sigma = np.full((nz, nx), sigma_background)

    center_j = 128
    width = 20
    for j in range(center_j - width, center_j + width):
        if 0 <= j < nx:
            sigma[:, j] = sigma_dike

    terrain = np.full(nx, 26)

    return sigma, terrain


def run_dike_model():
    """
    运行Dike模型正演
    """
    nx, nz = 256, 256
    pad = 20

    sigma, terrain = build_dike_model()

    frequencies = np.array([10000.0, 100000.0, 250000.0])

    dy = np.ones(nx) * 50.0
    dz = np.ones(nz) * 50.0

    print("Running dike model forward...")
    results = rfmt_forward(sigma, terrain, frequencies, dy, dz)

    print(f"Results shape: {results.shape}")

    save_results(results, 'dike_model_result.txt')
    print("Results saved to dike_model_result.txt")


if __name__ == '__main__':
    run_dike_model()

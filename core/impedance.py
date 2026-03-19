"""
Impedance and Apparent Resistivity Calculation
阻抗与视电阻率计算
"""

import numpy as np


def compute_te_impedance(Ex: np.ndarray, grid, terrain: np.ndarray,
                         omega: float, mu: float = np.pi * 4e-7) -> np.ndarray:
    """
    计算TE模式阻抗 Zxy = Ex / Hy

    对应Fortran lines 162-168:
    Hy1_2 = (Ex(n+cols+1)-Ex(n)+Ex(n+cols+2)-Ex(n+1))/dz/(i*ω*μ)/2
    Ex1_2 = (Ex(n+cols+1)+Ex(n)+Ex(n+cols+2)+Ex(n+1))/4
    Zxy = Ex1_2 / Hy1_2

    Args:
        Ex: 电场解向量
        grid: 网格对象
        terrain: 地形数组
        omega: 角频率 (rad/s)
        mu: 磁导率 (H/m)

    Returns:
        Zxy: 复阻抗数组
    """
    core = grid.core_cols
    Zxy = np.zeros(core, dtype=complex)

    for j in range(core):
        jn = j + grid.pad
        n = grid.node_index(jn, terrain[jn])

        Ex_avg = (Ex[n] + Ex[n + grid.nx + 1] +
                  Ex[n + grid.nx + 2] + Ex[n + 1]) / 4
        Hy = (Ex[n + grid.nx + 1] - Ex[n] +
               Ex[n + grid.nx + 2] - Ex[n + 1]) / grid.dz[terrain[jn] - 1] / (1j * omega * mu) / 2

        Zxy[j] = Ex_avg / Hy

    return np.conj(Zxy)


def compute_tm_impedance(Hx: np.ndarray, grid, terrain: np.ndarray,
                        omega: float, sigma: np.ndarray,
                        mu: float = np.pi * 4e-7,
                        eps0: float = 8.854e-12) -> np.ndarray:
    """
    计算TM模式阻抗 Zyx = Ey / Hx

    对应Fortran lines 312-325:
    Ey = (Hx(n+cols+1)-Hx(n))/dz/(σ-i*ω*ε)
    Hx = (Hx(n+cols+1)+Hx(n))/2
    Zyx = Ey / Hx

    Args:
        Hx: 磁场解向量
        grid: 网格对象
        terrain: 地形数组
        omega: 角频率 (rad/s)
        sigma: 电导率数组
        mu: 磁导率 (H/m)
        eps0: 真空介电常数 (F/m)

    Returns:
        Zyx: 复阻抗数组
    """
    core = grid.core_cols
    Zyx = np.zeros(core, dtype=complex)

    for j in range(core):
        jn = j + grid.pad
        n = grid.node_index(jn, terrain[jn])

        sig = sigma[jn, terrain[jn] - 1] - 1j * omega * eps0

        Ey = (Hx[n + grid.nx + 1] - Hx[n]) / grid.dz[terrain[jn] - 1] / sig
        Hx_avg = (Hx[n + grid.nx + 1] + Hx[n]) / 2

        Zyx[j] = Ey / Hx_avg

    return np.conj(Zyx)


def apparent_resistivity(Z: np.ndarray, omega: float,
                        mu: float = np.pi * 4e-7):
    """
    计算视电阻率和相位

    Args:
        Z: 复阻抗 (numpy.ndarray)
        omega: 角频率 (rad/s)
        mu: 磁导率 (H/m)

    Returns:
        rho_a: 视电阻率 (Ω·m)
        phi: 相位 (度)
    """
    rho_a = np.abs(Z) ** 2 / (omega * mu)
    phi = np.angle(Z, deg=True)
    return rho_a, phi

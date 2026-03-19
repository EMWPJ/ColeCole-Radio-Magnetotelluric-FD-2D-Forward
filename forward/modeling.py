"""
Main Forward Modeling Program
正演主程序
"""

import numpy as np

from ..core.cole_cole import ColeColeSingle
from ..core.mesh import RectGrid256
from ..core.fd_operator import assemble_te_matrix, assemble_tm_matrix
from ..core.boundary import apply_te_bc, apply_tm_bc
from ..core.solver import ComplexPardisoSolver
from ..core.impedance import (compute_te_impedance,
                              compute_tm_impedance,
                              apparent_resistivity)


def rfmt_forward(sigma: np.ndarray, terrain: np.ndarray,
                 frequencies: np.ndarray, dy: np.ndarray, dz: np.ndarray,
                 sigma0: float = 0.001, m: float = 0.1,
                 tau: float = 1e-3, c: float = 0.5,
                 epsilon_r: float = 5.0, pad: int = 20):
    """
    RMT二维Cole-Cole正演主程序

    Args:
        sigma: 256×256 电导率数组 (S/m)
        terrain: 256 地形数组（每列地表行号，1-indexed）
        frequencies: nf 频率数组 (Hz)
        dy: 256 变步长x方向 (m)
        dz: 256 变步长z方向 (m)
        sigma0: Cole-Cole静态电导率 (S/m)
        m: Cole-Cole极化强度
        tau: Cole-Cole时间常数 (s)
        c: Cole-Cole指数
        epsilon_r: 相对介电常数
        pad: 边界padding层数

    Returns:
        results: (nfreq * ncore_cols, 7) 数组
                [fre, y, z, rhoxy, rhoyx, phasexy, phaseyx]
    """
    grid = RectGrid256(nx=256, nz=256, pad=pad)
    mu = np.pi * 4e-7
    eps0 = 8.854e-12

    cc = ColeColeSingle(sigma0=sigma0, m=m, tau=tau, c=c)

    core_j = list(range(pad, 256 - pad))

    results = []

    for f in frequencies:
        omega = 2 * np.pi * f

        sigma_cc = cc.sigma_total(omega, epsilon_r)
        sigma_full = sigma_cc * np.ones_like(sigma)

        sigma_hat = grid.sigma_hat(sigma_full)

        A_te = assemble_te_matrix(grid, sigma_hat, omega, mu)
        b_te = np.zeros(grid.nnodes, dtype=complex)
        apply_te_bc(A_te, b_te, grid, Ex_value=1.0)

        solver_te = ComplexPardisoSolver(A_te)
        Ex = solver_te.solve(b_te).astype(complex)
        solver_te.free()

        Zxy = compute_te_impedance(Ex, grid, terrain, omega, mu)
        rho_xy, phi_xy = apparent_resistivity(Zxy, omega, mu)

        A_tm = assemble_tm_matrix(grid, sigma_hat, omega, mu, eps0)
        b_tm = np.zeros(grid.nnodes, dtype=complex)
        apply_tm_bc(A_tm, b_tm, grid, Hx_value=0.0)

        solver_tm = ComplexPardisoSolver(A_tm)
        Hx = solver_tm.solve(b_tm).astype(complex)
        solver_tm.free()

        Zyx = compute_tm_impedance(Hx, grid, terrain, omega, sigma, mu, eps0)
        rho_yx, phi_yx = apparent_resistivity(Zyx, omega, mu)

        for j in core_j:
            jn = j
            k = terrain[jn] - 1
            y = grid.y_nodes[jn]
            z = grid.z_nodes[k]

            j_idx = j - pad
            results.append([f, y, z,
                          rho_xy[j_idx], rho_yx[j_idx],
                          phi_xy[j_idx], phi_yx[j_idx]])

    return np.array(results)


def save_results(results: np.ndarray, filename: str = 'result_core.txt'):
    """
    保存核心区结果为文本文件

    Args:
        results: 结果数组
        filename: 输出文件名
    """
    header = ("fre(Hz)        y(m)           z(m)           "
              "rhoxy(Ω·m)   rhoyx(Ω·m)   phasexy(°)    phaseyx(°)")

    np.savetxt(filename, results,
                fmt='%.6e %.6e %.6e %.6e %.6e %.6e %.6e',
                header=header,
                comments='# ')

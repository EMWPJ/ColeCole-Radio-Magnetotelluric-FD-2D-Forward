"""
Finite Difference Operators for TE and TM Modes
有限差分算子 - TE和TM模式
"""

import numpy as np
from scipy.sparse import csr_matrix


def _add_sparse_entry(rows: list, cols: list, data: list,
                      r: int, c: int, v: complex):
    """添加稀疏矩阵元素"""
    rows.append(r)
    cols.append(c)
    data.append(v)


def assemble_te_matrix(grid, sigma_hat: np.ndarray,
                      omega: float, mu: float = np.pi * 4e-7) -> csr_matrix:
    """
    组装TE模式五点差分刚度矩阵

    对应Fortran MT2DTE子程序

    方程: (1/iωμ)∇²Eₓ + σ*Eₓ = 0

    Args:
        grid: RectGrid256网格对象
        sigma_hat: 变步长加权平均电导率
        omega: 角频率 (rad/s)
        mu: 磁导率 (H/m)

    Returns:
        刚度矩阵 (scipy.sparse.csr_matrix)
    """
    n = grid.nnodes
    rows, cols, data = [], [], []

    for k in range(1, grid.nz):
        for j in range(1, grid.nx):
            idx = grid.node_index(j, k)

            c_n = 2.0 / (grid.dz[k - 1] * (grid.dz[k - 1] + grid.dz[k]))
            c_s = 2.0 / (grid.dz[k] * (grid.dz[k - 1] + grid.dz[k]))
            c_w = 2.0 / (grid.dy[j - 1] * (grid.dy[j - 1] + grid.dy[j]))
            c_e = 2.0 / (grid.dy[j] * (grid.dy[j - 1] + grid.dy[j]))

            c_center = -(c_n + c_s + c_w + c_e) + 1j * omega * mu * sigma_hat[k, j]

            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j, k - 1), c_n)
            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j, k + 1), c_s)
            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j - 1, k), c_w)
            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j + 1, k), c_e)
            _add_sparse_entry(rows, cols, data, idx, idx, c_center)

    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)


def assemble_tm_matrix(grid, sigma_hat: np.ndarray,
                      omega: float, mu: float = np.pi * 4e-7,
                      eps0: float = 8.854e-12) -> csr_matrix:
    """
    组装TM模式五点差分刚度矩阵

    对应Fortran MT2DTM子程序

    方程: ∇·(1/σ*∇Hₓ) + iωμHₓ = 0

    Args:
        grid: RectGrid256网格对象
        sigma_hat: 变步长加权平均电导率
        omega: 角频率 (rad/s)
        mu: 磁导率 (H/m)
        eps0: 真空介电常数 (F/m)

    Returns:
        刚度矩阵 (scipy.sparse.csr_matrix)
    """
    n = grid.nnodes
    rows, cols, data = [], [], []

    sigma_y = np.zeros((grid.nz + 1, grid.nx + 1), dtype=complex)
    sigma_z = np.zeros((grid.nz + 1, grid.nx + 1), dtype=complex)

    for j in range(1, grid.nx):
        for k in range(1, grid.nz):
            sigma_y[k, j] = (sigma_hat[k, j] * grid.dz[k] +
                              sigma_hat[k - 1, j] * grid.dz[k - 1]) / (grid.dz[k] + grid.dz[k - 1])
            sigma_z[k, j] = (sigma_hat[k, j] * grid.dy[j] +
                              sigma_hat[k, j - 1] * grid.dy[j - 1]) / (grid.dy[j] + grid.dy[j - 1])

    for k in range(1, grid.nz):
        for j in range(1, grid.nx):
            idx = grid.node_index(j, k)

            coef_n = 2.0 / ((sigma_z[k - 1, j] - 1j * omega * eps0) *
                             grid.dz[k - 1] * (grid.dz[k - 1] + grid.dz[k]))
            coef_s = 2.0 / ((sigma_z[k, j] - 1j * omega * eps0) *
                             grid.dz[k] * (grid.dz[k - 1] + grid.dz[k]))
            coef_w = 2.0 / ((sigma_y[k, j - 1] - 1j * omega * eps0) *
                             grid.dy[j - 1] * (grid.dy[j - 1] + grid.dy[j]))
            coef_e = 2.0 / ((sigma_y[k, j] - 1j * omega * eps0) *
                             grid.dy[j] * (grid.dy[j - 1] + grid.dy[j]))

            c_center = -(coef_n + coef_s + coef_w + coef_e) + 1j * omega * mu

            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j, k - 1), coef_n)
            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j, k + 1), coef_s)
            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j - 1, k), coef_w)
            _add_sparse_entry(rows, cols, data, idx, grid.node_index(j + 1, k), coef_e)
            _add_sparse_entry(rows, cols, data, idx, idx, c_center)

    return csr_matrix((data, (rows, cols)), shape=(n, n), dtype=complex)

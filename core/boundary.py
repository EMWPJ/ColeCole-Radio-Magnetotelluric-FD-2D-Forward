"""
Boundary Conditions for TE and TM Modes
边界条件处理
"""

import numpy as np
from scipy.sparse import csr_matrix


def apply_te_bc(A, b, grid, Ex_value: float = 1.0):
    """
    施加TE模式边界条件

    对应Fortran lines 61-123:
    - 上边界: Ex = 1
    - 左边界: Ex(K,1) - Ex(K,2) = 0 (∂Ex/∂x = 0)
    - 右边界: Ex(K,nx) - Ex(K,nx+1) = 0
    - 下边界: ∂Ex/∂z = 0

    Args:
        A: 刚度矩阵
        b: 右端向量
        grid: 网格对象
        Ex_value: 上边界电场值
    """
    A = A.tolil()
    n = grid.nnodes

    for j in range(1, grid.nx):
        idx = grid.node_index(j, 0)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        b[idx] = Ex_value

    for k in range(1, grid.nz):
        idx = grid.node_index(0, k)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        A[idx, grid.node_index(1, k)] = -1.0
        b[idx] = 0.0

    for k in range(1, grid.nz):
        idx = grid.node_index(grid.nx, k)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        A[idx, grid.node_index(grid.nx - 1, k)] = -1.0
        b[idx] = 0.0

    for j in range(1, grid.nx):
        idx = grid.node_index(j, grid.nz)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        A[idx, grid.node_index(j, grid.nz - 1)] = -1.0
        b[idx] = 0.0

    A = A.tocsr()


def apply_tm_bc(A, b, grid, Hx_value: float = 0.0):
    """
    施加TM模式边界条件

    对应Fortran lines 208-270:
    - 上边界: Hx = 0
    - 左/右边界: ∂Hx/∂x = 0
    - 下边界: ∂Hx/∂z = 0

    Args:
        A: 刚度矩阵
        b: 右端向量
        grid: 网格对象
        Hx_value: 上边界磁场值
    """
    A = A.tolil()
    n = grid.nnodes

    for j in range(1, grid.nx):
        idx = grid.node_index(j, 0)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        b[idx] = Hx_value

    for k in range(1, grid.nz):
        idx = grid.node_index(0, k)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        A[idx, grid.node_index(1, k)] = -1.0
        b[idx] = 0.0

    for k in range(1, grid.nz):
        idx = grid.node_index(grid.nx, k)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        A[idx, grid.node_index(grid.nx - 1, k)] = -1.0
        b[idx] = 0.0

    for j in range(1, grid.nx):
        idx = grid.node_index(j, grid.nz)
        A[idx, :] = 0
        A[idx, idx] = 1.0
        A[idx, grid.node_index(j, grid.nz - 1)] = -1.0
        b[idx] = 0.0

    A = A.tocsr()

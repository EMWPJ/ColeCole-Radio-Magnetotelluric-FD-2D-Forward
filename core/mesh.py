"""
Rectangular Grid Module for 256x256 Mesh
256×256矩形网格模块
"""

import numpy as np


class RectGrid256:
    """
    256×256非交错矩形网格

    节点编号: (j, k) -> n = k * (nx+1) + j
    电导率数组: sigma[k, j] 对应 sigma(j, k) in Fortran

    Attributes:
        nx: 列数 (256)
        nz: 行数 (256)
        pad: 边界padding层数 (20)
        core_cols: 核心区列数 (216)
        nnodes: 总节点数 (257×257 = 66049)
        dy: x方向变步长数组
        dz: z方向变步长数组
        y_nodes: y方向节点坐标
        z_nodes: z方向节点坐标
    """

    def __init__(self, nx: int = 256, nz: int = 256, pad: int = 20):
        """
        初始化网格

        Args:
            nx: 列数
            nz: 行数
            pad: 边界padding层数
        """
        self.nx = nx
        self.nz = nz
        self.pad = pad
        self.core_cols = nx - 2 * pad
        self.nnodes = (nz + 1) * (nx + 1)

        self.dy = self._generate_dy()
        self.dz = self._generate_dz()

        self.y_nodes = np.concatenate([[0], np.cumsum(self.dy)])
        self.z_nodes = np.concatenate([[0], np.cumsum(self.dz)])

    def _generate_dy(self) -> np.ndarray:
        """
        y方向变步长：中间密、两端疏

        Returns:
            dy数组 shape (nx,)
        """
        dy = np.zeros(self.nx)
        center_start = self.pad
        center_end = self.nx - self.pad
        center_dy = 50.0

        for j in range(self.nx):
            if j < center_start:
                factor = 1.0 + (center_start - j) * 0.15
                dy[j] = center_dy * factor
            elif j > center_end:
                factor = 1.0 + (j - center_end) * 0.15
                dy[j] = center_dy * factor
            else:
                dy[j] = center_dy

        return dy

    def _generate_dz(self) -> np.ndarray:
        """
        z方向变步长：浅层密、深层疏

        Returns:
            dz数组 shape (nz,)
        """
        dz = np.zeros(self.nz)
        dz0 = 20.0
        growth = 1.04

        for k in range(self.nz):
            dz[k] = dz0 * (growth ** k)

        total = np.sum(dz)
        dz = dz / total * 10000.0

        return dz

    def node_index(self, j: int, k: int) -> int:
        """
        将二维索引(j, k)转为一维节点编号

        Args:
            j: 列索引 (0 ~ nx)
            k: 行索引 (0 ~ nz)

        Returns:
            一维节点编号
        """
        return k * (self.nx + 1) + j

    def sigma_hat(self, sigma: np.ndarray) -> np.ndarray:
        """
        变步长加权平均电导率

        对应Fortran lines 50-58:
        sigma_hat(J1,K1) = weighted_average(sigma, dy, dz)

        Args:
            sigma: 电导率数组 shape (nz, nx)

        Returns:
            sigma_hat数组 shape (nz+1, nx+1)
        """
        sh = np.zeros((self.nz + 1, self.nx + 1), dtype=complex)

        sh[0, :] = sigma[0, :]
        sh[self.nz, :] = sigma[self.nz - 1, :]
        sh[:, 0] = sigma[:, 0]
        sh[:, self.nx] = sigma[:, self.nx - 1]

        for j in range(1, self.nx):
            for k in range(1, self.nz):
                w1 = self.dy[j - 1] * self.dz[k - 1]
                w2 = self.dy[j] * self.dz[k - 1]
                w3 = self.dy[j - 1] * self.dz[k]
                w4 = self.dy[j] * self.dz[k]
                wt = w1 + w2 + w3 + w4

                sh[k, j] = (sigma[k - 1, j - 1] * w1 +
                            sigma[k - 1, j] * w2 +
                            sigma[k, j - 1] * w3 +
                            sigma[k, j] * w4) / wt

        return sh

    def get_core_indices(self) -> list:
        """
        获取核心区列索引

        Returns:
            核心区列索引列表 [pad, ..., nx-pad-1]
        """
        return list(range(self.pad, self.nx - self.pad))

    def get_core_y_coords(self) -> np.ndarray:
        """
        获取核心区y坐标

        Returns:
            y坐标数组 shape (core_cols,)
        """
        return self.y_nodes[self.pad:self.nx - self.pad]

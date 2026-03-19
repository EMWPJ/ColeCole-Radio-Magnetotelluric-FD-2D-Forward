"""
Cole-Cole Model Implementation
单Cole-Cole极化模型
"""

import numpy as np


class ColeColeSingle:
    """
    单Cole-Cole极化模型

    sigma*(omega) = sigma0 * [1 + m * (1 - 1/(1 + (i*omega*tau)^c))]

    Attributes:
        sigma0: 静态电导率 (S/m)
        m: 极化强度 (0~1)
        tau: 时间常数 (秒)
        c: Cole-Cole指数 (0~1)
    """

    def __init__(self, sigma0: float, m: float, tau: float, c: float):
        """
        初始化Cole-Cole模型

        Args:
            sigma0: 静态电导率 (S/m)
            m: 极化强度 (0~1)
            tau: 时间常数 (秒)
            c: Cole-Cole指数 (0~1)
        """
        self.sigma0 = sigma0
        self.m = m
        self.tau = tau
        self.c = c

    def sigma(self, omega: float) -> complex:
        """
        计算Cole-Cole复电导率

        Args:
            omega: 角频率 (rad/s)

        Returns:
            复电导率 (S/m)
        """
        v = (1j * omega * self.tau) ** self.c
        denom = 1.0 + v
        factor = 1.0 + self.m * (1.0 - 1.0 / denom)
        return self.sigma0 * factor

    def sigma_total(self, omega: float, epsilon_r: float = 1.0) -> complex:
        """
        含位移电流的总电导率

        Args:
            omega: 角频率 (rad/s)
            epsilon_r: 相对介电常数

        Returns:
            含位移电流的总电导率 (S/m)
        """
        eps0 = 8.854e-12
        return self.sigma(omega) + 1j * omega * eps0 * epsilon_r


def cole_cole_conductivity(sigma0: float, m: float, tau: float, c: float,
                          omega: float, epsilon_r: float = 1.0) -> complex:
    """
    便捷函数：计算Cole-Cole复电导率（含位移电流）

    Args:
        sigma0: 静态电导率 (S/m)
        m: 极化强度
        tau: 时间常数 (s)
        c: Cole-Cole指数
        omega: 角频率 (rad/s)
        epsilon_r: 相对介电常数

    Returns:
        总电导率 (S/m)
    """
    cc = ColeColeSingle(sigma0, m, tau, c)
    return cc.sigma_total(omega, epsilon_r)

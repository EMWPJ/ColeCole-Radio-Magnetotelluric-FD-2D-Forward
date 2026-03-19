"""
Pardiso Solver Wrapper
pypardiso求解器封装
"""

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve


class ComplexPardisoSolver:
    """
    pypardiso复数矩阵求解器

    封装Intel MKL PARDISO求解器

    Attributes:
        A: 系数矩阵
        solver: 预分解的求解器
    """

    def __init__(self, A):
        """
        初始化求解器

        Args:
            A: 稀疏矩阵 (scipy.sparse.csr_matrix)
        """
        self.A = csc_matrix(A)

        try:
            from pypardiso.scipy_solver import PardisoSPSolver
            self.solver = PardisoSPSolver(self.A, factorized=True)
            self.use_pardiso = True
        except ImportError:
            self.use_pardiso = False

    def solve(self, b):
        """
        求解 Ax = b

        Args:
            b: 右端向量 (numpy.ndarray)

        Returns:
            x: 解向量 (numpy.ndarray)
        """
        if self.use_pardiso:
            return self.solver.solve(b)
        else:
            return spsolve(self.A, b)

    def free(self):
        """释放内存"""
        if self.use_pardiso and hasattr(self, 'solver'):
            del self.solver

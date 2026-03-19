# ColeCole-Radio-Magnetotelluric-FD-2D-Forward

RMT (Radio Magnetotelluric) 2D Finite Difference Forward Modeling with Cole-Cole Dispersion Model

射频大地电磁法二维有限差分正演模拟程序 - 含Cole-Cole色散模型

## 项目简介

本程序实现RMT二维正演模拟，包含激发极化（IP）效应的Cole-Cole色散模型。基于Fortran参考实现开发，用于研究激发极化效应对射频大地电磁响应的影响。

**参考论文：**
- 原源等. 基于非结构化网格的任意复杂2DRMT有限元模拟[J]. 地球物理学报, 2015, 58(12): 4685-4695.
- 王培杰等. 基于四叉树网格的MT二维正演[J]. 石油地球物理勘探, 2019, 54(3): 709-718.

## 目录结构

```
rfmt_cole_cole/
├── core/                    # 核心模块
│   ├── cole_cole.py         # Cole-Cole模型
│   ├── mesh.py              # 256×256矩形网格
│   ├── fd_operator.py       # TE/TM模式有限差分算子
│   ├── boundary.py          # 边界条件
│   ├── solver.py            # pypardiso求解器封装
│   └── impedance.py         # 阻抗与视电阻率计算
├── forward/
│   └── modeling.py          # 正演主程序
├── tests/
│   └── test_uniform.py      # 均匀半空间验证测试
├── examples/
│   ├── dike_model.py        # 岩脉模型示例
│   └── custom_cole_cole.py  # 自定义参数示例
├── docs/
│   └── 开发文档.md          # 详细设计文档
├── config.py                # 配置参数
├── rfmt_fwd.py             # 程序入口
└── requirements.txt        # 依赖包
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `numpy` - 数值计算
- `scipy` - 科学计算
- `pypardiso` - Intel MKL PARDISO求解器（可选，默认使用scipy.spsolve）

### 2. 运行测试

```bash
cd tests
python test_uniform.py
```

这将验证均匀半空间模型的正确性，解析解为1000 Ω·m。

### 3. 运行示例

```bash
python rfmt_fwd.py
```

## 使用说明

### 基本用法

```python
import numpy as np
from forward.modeling import rfmt_forward, save_results

# 模型参数
nx, nz = 256, 256
sigma = np.full((nz, nx), 0.001)  # 电导率 (S/m)
terrain = np.full(nx, 26)         # 地形数组（每列地表行号）

# 频率范围 (Hz)
frequencies = np.logspace(4, 5.4, 5)

# 网格步长 (m)
dy = np.ones(nx) * 50.0
dz = np.ones(nz) * 50.0

# Cole-Cole参数
sigma0 = 0.001   # 静态电导率 (S/m)
m = 0.1          # 极化强度 (0~1)
tau = 1e-3       # 时间常数 (s)
c = 0.5          # Cole-Cole指数 (0~1)
epsilon_r = 5.0  # 相对介电常数

# 运行正演
results = rfmt_forward(
    sigma, terrain, frequencies, dy, dz,
    sigma0=sigma0, m=m, tau=tau, c=c, epsilon_r=epsilon_r
)

# 保存结果
save_results(results, 'result_core.txt')
```

### Cole-Cole模型说明

```
sigma*(ω) = sigma0 × [1 + m × (1 - 1/(1 + (i×ω×τ)^c))]
```

参数含义：
- `sigma0` - 静态电导率（低频极限）
- `m` - 极化强度（0~1），表示IP效应强弱
- `tau` - 时间常数（秒），与充放电过程相关
- `c` - Cole-Cole指数（0~1），描述色散关系的频率分布

## 输出格式

结果文件包含7列数据：
```
fre(Hz)    y(m)       z(m)       rhoxy(Ω·m)  rhoyx(Ω·m)  phasexy(°)  phaseyx(°)
```

- `fre` - 频率 (Hz)
- `y` - 水平位置 (m)
- `z` - 深度 (m)
- `rhoxy` - TE模式视电阻率
- `rhoyx` - TM模式视电阻率
- `phasexy` - TE模式相位 (度)
- `phaseyx` - TM模式相位 (度)

## 下一步开发指南

### 1. 验证与对比

- [ ] **对比Fortran结果**：将Python输出与`fort.123`对比
- [ ] **均匀半空间测试**：验证解析解（rho = 1000 Ω·m）
- [ ] **参考论文验证**：对比Yuan et al. 2015 Figure 4, 5

### 2. 功能扩展

- [ ] **变间距网格**：实现与Fortran一致的变步长（当前为均匀50m）
- [ ] **非均匀模型**：从`sigma.txt`读取真实电导率模型
- [ ] **地形校正**：支持任意地形数组
- [ ] **多极Cole-Cole**：扩展到多极模型

### 3. 性能优化

- [ ] **向量化阻抗计算**：避免循环，提升计算效率
- [ ] **GPU加速**：使用CuPy或 JAX GPU后端
- [ ] **并行频率计算**：多频率并行正演

### 4. 后处理与可视化

- [ ] **拟断面图**：绘制ρa和φ拟断面
- [ ] **曲线对比**：多模型曲线对比
- [ ] **误差分析**：与解析解/参考解的误差分布

### 5. 测试用例

- [ ] **层状模型**：添加层状模型解析解
- [ ] **异常体模型**：岩脉、薄板等简单模型
- [ ] **边界条件测试**：验证各种边界条件
- [ ] **Cole-Cole参数敏感性分析**

## 与Fortran代码对应关系

| Fortran | Python |
|---------|--------|
| MT2DFwd.f90 | forward/modeling.py |
| MT2DTE子程序 | core/fd_operator.py (assemble_te_matrix) |
| MT2DTM子程序 | core/fd_operator.py (assemble_tm_matrix) |
| sigma.txt | sigma数组 |
| terrain.txt | terrain数组 |
| fort.123 | result_core.txt |

## 参考资源

### 理论背景
- Yuan et al. 2015 - RMT有限元模拟位移电流
- Wang et al. 2019 - 四叉树网格MT二维正演
- Wait 1954 - 极化介质的电磁响应理论

### 代码参考
- Fortran MT2D参考实现：`D:/南科大研究/MT2D/MT2D2025/`

## 注意事项

1. **网格索引**：Python使用0-indexing，Fortran使用1-indexing
2. **坐标系**：y向右为正，z向下为正
3. **电导率数组**：`sigma[k, j]` 对应 Fortran中的 `sigma(j, k)`
4. **节点编号**：`node_index(j, k) = k * (nx+1) + j`

## 许可证

本项目仅供学术研究使用。

## 联系方式

如有问题或建议，请提交Issue。

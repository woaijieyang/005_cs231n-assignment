import numpy as np

"""
此文件执行常用的各种第一阶更新规则
训练神经网络。每个更新规则都接受当前重量和
这些重量损失的梯度,并产生下一组
权重。每个更新规则的接口相同 :

def 更新( w, dw, config=无) :

输入：
  - w: 给出当前权重的数组 。
  - dw: 与 w 的形状相同的 numpy 数组, 给出渐变
    关于W. 损失
  - config: 包含超参数值的字典,如学习率,
    如果更新规则要求缓存值超过许多
    迭代,然后配置也会持有这些缓存值.

返回：
  - next_w: 更新后的下一点.
  - config: 配置词典要传递到下一个迭代
    更新规则。

NOTE: 对大多数更新规则来说,默认学习率可能不会实现
但是其他超参数的默认值应该很好使用
对于各种各样的不同问题。

为提高效率,更新规则可在现场进行更新,使w和
设置下一个_w 等于w。
"""


# 输入当前参数、同形状梯度及学习率，沿负梯度更新。返回新参数和配置，某些实现原地修改输入参数，调用方不能假定原数组仍是旧值。
def sgd(w, dw, config=None):
    """输入当前参数、同形状梯度及学习率，沿负梯度更新。返回新参数和配置，某些实现原地修改输入参数，调用方不能假定原数组仍是旧值。

    参数在本函数中的角色：
    - w：本层的权重。
    - config：该参数独立的超参数与历史状态。

    返回表达式：(w, config)。
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


# 分别更新一阶矩 m 和二阶矩 v，递增时间步并进行初值偏差校正，再更新参数。Assignment 2 使用标准校正矩写法；Assignment 3 保留课程 alpha/epsilon 位置，GPU 对照也保持同一公式。
def adam(x, dx, config=None):
    """分别更新一阶矩 m 和二阶矩 v，递增时间步并进行初值偏差校正，再更新参数。Assignment 2 使用标准校正矩写法；Assignment 3 保留课程 alpha/epsilon 位置，GPU 对照也保持同一公式。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - config：该参数独立的超参数与历史状态。

    返回表达式：(next_x, config)。
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(x))
    config.setdefault('v', np.zeros_like(x))
    config.setdefault('t', 0)

    next_x = None
    beta1, beta2, eps = config['beta1'], config['beta2'], config['epsilon']
    t, m, v = config['t'], config['m'], config['v']
    m = beta1 * m + (1 - beta1) * dx
    v = beta2 * v + (1 - beta2) * (dx * dx)
    t += 1
    alpha = config['learning_rate'] * np.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
    x -= alpha * (m / (np.sqrt(v) + eps))
    config['t'] = t
    config['m'] = m
    config['v'] = v
    next_x = x

    return next_x, config

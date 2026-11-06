import numpy as np

"""
此文件执行常用的各种第一阶更新规则
用于训练神经网络。每个更新规则都接受当前加权和
这些重量损失的梯度,并产生下一组
权重。每个更新规则的接口相同 :

def 更新( w, dw, config=无) :

输入：
  - w: 给出当前权重的数组 。
  - dw: 与 w 的形状相同的 numpy 数组, 给出渐变
    关于W. 损失
  - config: 包含超参数值的字典,如学习
    速率、动量等。如果更新规则要求缓存值超过许多
    迭代,然后配置也会持有这些缓存值.

返回：
  - next_w: 更新后的下一点.
  - config: 配置词典要传递到下一个迭代
    更新规则。

NOTE: 对于大多数更新规则,默认学习率可能不会
运行良好;但其他超参数的默认值应当
解决各种不同问题的工作效果良好。

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


# 更新速度 v=μv−ηdw，再令 w=w+v；每组参数独立保存速度。输入梯度与速度同形状，返回更新后的权重及带新速度的配置。
def sgd_momentum(w, dw, config=None):
    """更新速度 v=μv−ηdw，再令 w=w+v；每组参数独立保存速度。输入梯度与速度同形状，返回更新后的权重及带新速度的配置。

    参数在本函数中的角色：
    - w：本层的权重。
    - config：该参数独立的超参数与历史状态。

    返回表达式：(next_w, config)。
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    config['velocity'] = v

    return next_w, config


# 缓存梯度平方的指数滑动平均，以 sqrt(cache)+eps 调整各坐标步长。不要把本步梯度平方直接代替历史缓存；输出参数及更新后的 cache。
def rmsprop(x, dx, config=None):
    """缓存梯度平方的指数滑动平均，以 sqrt(cache)+eps 调整各坐标步长。不要把本步梯度平方直接代替历史缓存；输出参数及更新后的 cache。

    参数在本函数中的角色：
    - x：当前可训练参数，形状与 dx 一致；不是一批输入图像。
    - config：该参数独立的超参数与历史状态。

    返回表达式：(next_x, config)。
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx ** 2
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])

    return next_x, config


# 分别更新一阶矩 m 和二阶矩 v，递增时间步并进行初值偏差校正，再更新参数。Assignment 2 使用标准校正矩写法；Assignment 3 保留课程 alpha/epsilon 位置，GPU 对照也保持同一公式。
def adam(x, dx, config=None):
    """分别更新一阶矩 m 和二阶矩 v，递增时间步并进行初值偏差校正，再更新参数。Assignment 2 使用标准校正矩写法；Assignment 3 保留课程 alpha/epsilon 位置，GPU 对照也保持同一公式。

    参数在本函数中的角色：
    - x：当前可训练参数，将由一阶、二阶矩共同决定更新步长。
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
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx  # 一阶矩平滑当前梯度，保留方向信息；它属于当前参数，不能在不同权重间共享。
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx ** 2)  # 二阶矩平滑梯度平方，为不同坐标提供不同的步长尺度。
    # m/v 初值为零会造成初期低估；先递增时间步，再分别进行偏差校正。
    config['t'] += 1  # 这是该参数的更新次数。先从零递增，再用它计算一阶、二阶矩的初值偏差校正。
    m_hat = config['m'] / (1 - config['beta1'] ** config['t'])  # 一阶矩初值为零，除以 1−β₁ᵗ 校正前几次更新中均值被压小的效应。
    v_hat = config['v'] / (1 - config['beta2'] ** config['t'])  # 二阶矩同样进行初值校正；随后开平方得到逐坐标的梯度尺度。
    next_x = x - config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])

    return next_x, config

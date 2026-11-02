import numpy as np

"""
This file implements various first-order update rules that are commonly used
for training neural networks. Each update rule accepts current weights and the
gradient of the loss with respect to those weights and produces the next set of
weights. Each update rule has the same interface:

def update(w, dw, config=None):

Inputs:
  - w: A numpy array giving the current weights.
  - dw: A numpy array of the same shape as w giving the gradient of the
    loss with respect to w.
  - config: A dictionary containing hyperparameter values such as learning
    rate, momentum, etc. If the update rule requires caching values over many
    iterations, then config will also hold these cached values.

Returns:
  - next_w: The next point after the update.
  - config: The config dictionary to be passed to the next iteration of the
    update rule.

NOTE: For most update rules, the default learning rate will probably not
perform well; however the default values of the other hyperparameters should
work well for a variety of different problems.

For efficiency, update rules may perform in-place updates, mutating w and
setting next_w equal to w.
"""


# 中文数据流：输入当前参数、同形状梯度及学习率，沿负梯度更新。返回新参数和配置，某些实现原地修改输入参数，调用方不能假定原数组仍是旧值。
def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


# 中文数据流：更新速度 v=μv−ηdw，再令 w=w+v；每组参数独立保存速度。输入梯度与速度同形状，返回更新后的权重及带新速度的配置。
def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the momentum update formula. Store the updated value in #
    # the next_w variable. You should also use and update the velocity v.     #
    ###########################################################################
    v = config['momentum'] * v - config['learning_rate'] * dw
    next_w = w + v
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    config['velocity'] = v

    return next_w, config


# 中文数据流：缓存梯度平方的指数滑动平均，以 sqrt(cache)+eps 调整各坐标步长。不要把本步梯度平方直接代替历史缓存；输出参数及更新后的 cache。
def rmsprop(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(x))

    next_x = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of x #
    # in the next_x variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    config['cache'] = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dx ** 2
    next_x = x - config['learning_rate'] * dx / (np.sqrt(config['cache']) + config['epsilon'])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config


# 中文数据流：分别更新一阶矩 m 和二阶矩 v，递增时间步并进行初值偏差校正，再更新参数。Assignment 2 使用标准校正矩写法；Assignment 3 保留课程 alpha/epsilon 位置，GPU 对照也保持同一公式。
def adam(x, dx, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
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
    ###########################################################################
    # TODO: Implement the Adam update formula, storing the next value of x in #
    # the next_x variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    ###########################################################################
    config['m'] = config['beta1'] * config['m'] + (1 - config['beta1']) * dx
    config['v'] = config['beta2'] * config['v'] + (1 - config['beta2']) * (dx ** 2)
    # m/v 初值为零会造成初期低估；先递增时间步，再分别进行偏差校正。
    config['t'] += 1
    m_hat = config['m'] / (1 - config['beta1'] ** config['t'])
    v_hat = config['v'] / (1 - config['beta2'] ** config['t'])
    next_x = x - config['learning_rate'] * m_hat / (np.sqrt(v_hat) + config['epsilon'])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_x, config

from cs231n.layers import *
from cs231n.fast_layers import *


# 按 affine → relu 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。
def affine_relu_forward(x, w, b):
    """按 affine → relu 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - w：本层的权重。
    - b：与输出维对齐的偏置。

    返回表达式：(out, cache)。
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


# 读取 affine_relu 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。
def affine_relu_backward(dout, cache):
    """读取 affine_relu 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


# 按 affine → bn → relu 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。
def affine_bn_relu_forward(x, w, b, gamma, beta, bn_param):
    """按 affine → bn → relu 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - w：本层的权重。
    - b：与输出维对齐的偏置。
    - gamma：可训练的逐特征缩放系数。
    - beta：可训练的逐特征偏移量。
    - bn_param：模式、数值稳定常数和滑动统计等配置。

    返回表达式：(out, cache)。
    """
    a, fc_cache = affine_forward(x, w, b)
    a_bn, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(a_bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache


# 读取 affine_bn_relu 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。
def affine_bn_relu_backward(dout, cache):
    """读取 affine_bn_relu 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db, dgamma, dbeta)。
    """
    fc_cache, bn_cache, relu_cache = cache
    da_bn = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = batchnorm_backward(da_bn, bn_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db, dgamma, dbeta


# 按 conv → relu 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。
def conv_relu_forward(x, w, b, conv_param):
    """按 conv → relu 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - w：本层的权重。
    - b：与输出维对齐的偏置。
    - conv_param：卷积窗口步幅与边界填充。

    返回表达式：(out, cache)。
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


# 读取 conv_relu 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。
def conv_relu_backward(dout, cache):
    """读取 conv_relu 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db


# 按 conv → bn → relu 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。
def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


# 读取 conv_bn_relu 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。
def conv_bn_relu_backward(dout, cache):
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


# 按 conv → relu → pool 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。
def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """按 conv → relu → pool 顺序调用基础层，将各层 cache 按相同顺序保存。输出最终激活及组合 cache；该包装不额外引入训练参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - w：本层的权重。
    - b：与输出维对齐的偏置。
    - conv_param：卷积窗口步幅与边界填充。
    - pool_param：池化窗口大小与步幅。

    返回表达式：(out, cache)。
    """
    a, conv_cache = conv_forward_fast(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward_fast(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


# 读取 conv_relu_pool 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。
def conv_relu_pool_backward(dout, cache):
    """读取 conv_relu_pool 前向保存的各层 cache，按计算顺序的逆序传播 dout。返回输入梯度和各参数梯度，具体形状与各基础层接口一致。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_fast(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_fast(da, conv_cache)
    return dx, dw, db

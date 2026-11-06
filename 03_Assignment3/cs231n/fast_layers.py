from __future__ import print_function
import numpy as np

try:
    from cs231n.im2col_cython import col2im_cython, im2col_cython
    from cs231n.im2col_cython import col2im_6d_cython
except ImportError:
    print('run the following from the cs231n directory and try again:')
    print('python setup.py build_ext --inplace')
    print('You may also need to restart your iPython kernel')

from cs231n.im2col import *


# 将局部卷积窗口重排为列，以矩阵乘法同时计算所有核和位置，再恢复 NCHW 输出。cache 保存窗口排列，反向必须按同一索引规则聚合重叠贡献。
def conv_forward_im2col(x, w, b, conv_param):
    """将局部卷积窗口重排为列，以矩阵乘法同时计算所有核和位置，再恢复 NCHW 输出。cache 保存窗口排列，反向必须按同一索引规则聚合重叠贡献。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - w：本层的权重。
    - b：与输出维对齐的偏置。
    - conv_param：卷积窗口步幅与边界填充。

    返回表达式：(out, cache)。
    """
    N, C, H, W = x.shape
    num_filters, _, filter_height, filter_width = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
    assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

    out_height = (H + 2 * pad - filter_height) // stride + 1
    out_width = (W + 2 * pad - filter_width) // stride + 1
    out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

    x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)  # 把每个局部窗口排成一列，一行对应一个通道和核内坐标，便于下一步与展平的卷积核相乘。
    res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

    out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


# 将局部卷积窗口重排为列，以矩阵乘法同时计算所有核和位置，再恢复 NCHW 输出。cache 保存窗口排列，反向必须按同一索引规则聚合重叠贡献。
def conv_forward_strides(x, w, b, conv_param):
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']


    p = pad
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    H += 2 * pad
    W += 2 * pad
    out_h = (H - HH) // stride + 1
    out_w = (W - WW) // stride + 1

    shape = (C, HH, WW, N, out_h, out_w)
    strides = (H * W, W, 1, C * H * W, stride * W, stride)
    strides = x.itemsize * np.array(strides)  # as_strided 的步长单位是字节，因此这里把“跨几个元素”乘以单个元素的字节数。
    x_stride = np.lib.stride_tricks.as_strided(x_padded,
                                               shape=shape, strides=strides)
    x_cols = np.ascontiguousarray(x_stride)
    x_cols.shape = (C * HH * WW, N * out_h * out_w)  # 行数是一个窗口的元素数；列数是所有图片的所有输出位置数。矩阵乘积的每列就是一个窗口的滤波结果。

    res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

    res.shape = (F, N, out_h, out_w)
    out = res.transpose(1, 0, 2, 3)

    out = np.ascontiguousarray(out)

    cache = (x, w, b, conv_param, x_cols)
    return out, cache


# 将 dout 转成与前向列矩阵一致的布局，用矩阵乘法求 dW 和窗口梯度，再执行 col2im 累加回原输入。db 沿 batch 和空间轴求和。
def conv_backward_strides(dout, cache):
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, out_h, out_w = dout.shape

    db = np.sum(dout, axis=(0, 2, 3))

    dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)  # 将输出梯度乘卷积核的转置，先得到每个窗口内部各位置的梯度，随后还需映射回原图。
    dx_cols.shape = (C, HH, WW, N, out_h, out_w)
    dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

    return dx, dw, db


# 将 dout 转成与前向列矩阵一致的布局，用矩阵乘法求 dW 和窗口梯度，再执行 col2im 累加回原输入。db 沿 batch 和空间轴求和。
def conv_backward_im2col(dout, cache):
    """将 dout 转成与前向列矩阵一致的布局，用矩阵乘法求 dW 和窗口梯度，再执行 col2im 累加回原输入。db 沿 batch 和空间轴求和。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    x, w, b, conv_param, x_cols = cache
    stride, pad = conv_param['stride'], conv_param['pad']

    db = np.sum(dout, axis=(0, 2, 3))

    num_filters, _, filter_height, filter_width = w.shape
    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
    dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

    dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)  # 将输出梯度乘卷积核的转置，先得到每个窗口内部各位置的梯度，随后还需映射回原图。
    dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                       filter_height, filter_width, pad, stride)

    return dx, dw, db


conv_forward_fast = conv_forward_strides
conv_backward_fast = conv_backward_strides


# 按池化窗口与步幅选择快速排列方式，输出局部最大值并保留方法标记和 cache。reshape 路径仅适用于可整齐划分的窗口，其他情况使用展开路径。
def max_pool_forward_fast(x, pool_param):
    """按池化窗口与步幅选择快速排列方式，输出局部最大值并保留方法标记和 cache。reshape 路径仅适用于可整齐划分的窗口，其他情况使用展开路径。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - pool_param：池化窗口大小与步幅。

    返回表达式：(out, cache)。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ('reshape', reshape_cache)
    else:
        out, im2col_cache = max_pool_forward_im2col(x, pool_param)
        cache = ('im2col', im2col_cache)
    return out, cache


# 按前向的方法标记恢复窗口布局，把初始版本梯度送回最大值位置并还原输入形状。并列最大值分配采用此快速实现约定，随机无并列测试应与朴素实现一致。
def max_pool_backward_fast(dout, cache):
    """按前向的方法标记恢复窗口布局，把初始版本梯度送回最大值位置并还原输入形状。并列最大值分配采用此快速实现约定，随机无并列测试应与朴素实现一致。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：max_pool_backward_reshape(dout, real_cache)；max_pool_backward_im2col(dout, real_cache)。
    """
    method, real_cache = cache
    if method == 'reshape':
        return max_pool_backward_reshape(dout, real_cache)
    elif method == 'im2col':
        return max_pool_backward_im2col(dout, real_cache)
    else:
        raise ValueError('Unrecognized method "%s"' % method)


# 按池化窗口与步幅选择快速排列方式，输出局部最大值并保留方法标记和 cache。reshape 路径仅适用于可整齐划分的窗口，其他情况使用展开路径。
def max_pool_forward_reshape(x, pool_param):
    """按池化窗口与步幅选择快速排列方式，输出局部最大值并保留方法标记和 cache。reshape 路径仅适用于可整齐划分的窗口，其他情况使用展开路径。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - pool_param：池化窗口大小与步幅。

    返回表达式：(out, cache)。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(N, C, H // pool_height, pool_height,
                           W // pool_width, pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


# 按前向的方法标记恢复窗口布局，把初始版本梯度送回最大值位置并还原输入形状。并列最大值分配采用此快速实现约定，随机无并列测试应与朴素实现一致。
def max_pool_backward_reshape(dout, cache):
    """按前向的方法标记恢复窗口布局，把初始版本梯度送回最大值位置并还原输入形状。并列最大值分配采用此快速实现约定，随机无并列测试应与朴素实现一致。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：dx。
    """
    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx


# 按池化窗口与步幅选择快速排列方式，输出局部最大值并保留方法标记和 cache。reshape 路径仅适用于可整齐划分的窗口，其他情况使用展开路径。
def max_pool_forward_im2col(x, pool_param):
    """按池化窗口与步幅选择快速排列方式，输出局部最大值并保留方法标记和 cache。reshape 路径仅适用于可整齐划分的窗口，其他情况使用展开路径。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - pool_param：池化窗口大小与步幅。

    返回表达式：(out, cache)。
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    assert (H - pool_height) % stride == 0, 'Invalid height'
    assert (W - pool_width) % stride == 0, 'Invalid width'

    out_height = (H - pool_height) // stride + 1
    out_width = (W - pool_width) // stride + 1

    x_split = x.reshape(N * C, 1, H, W)
    x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
    x_cols_argmax = np.argmax(x_cols, axis=0)  # 沿每个池化窗口寻找最大值在窗口内的位置，反向会用这个位置决定梯度去向。
    x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
    out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

    cache = (x, x_cols, x_cols_argmax, pool_param)
    return out, cache


# 按前向的方法标记恢复窗口布局，把初始版本梯度送回最大值位置并还原输入形状。并列最大值分配采用此快速实现约定，随机无并列测试应与朴素实现一致。
def max_pool_backward_im2col(dout, cache):
    """按前向的方法标记恢复窗口布局，把初始版本梯度送回最大值位置并还原输入形状。并列最大值分配采用此快速实现约定，随机无并列测试应与朴素实现一致。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：dx。
    """
    x, x_cols, x_cols_argmax, pool_param = cache
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']

    dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
    dx_cols = np.zeros_like(x_cols)
    dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped  # 每个窗口的上游梯度只写给该窗口的最大值，其他位置保持为零。
    dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width,
                        padding=0, stride=stride)
    dx = dx.reshape(x.shape)

    return dx

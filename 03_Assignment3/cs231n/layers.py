import numpy as np


# 输入 x 的首维是样本数 N，其余维展平为 D；计算 (N,D)@(D,M)+(M,) 得到 (N,M)。缓存原 x、w、b，供反向恢复输入形状。
def affine_forward(x, w, b):
    """输入 x 的首维是样本数 N，其余维展平为 D；计算 (N,D)@(D,M)+(M,) 得到 (N,M)。缓存原 x、w、b，供反向恢复输入形状。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - w：本层的权重。
    - b：与输出维对齐的偏置。

    返回表达式：(out, cache)。
    """
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


# 输入为 dout(N,M) 和前向缓存。通过 dout@w.T 得到输入梯度并恢复原形状；展平输入转置乘 dout 得到 dW，沿样本轴求和得到 db。
def affine_backward(dout, cache):
    """输入为 dout(N,M) 和前向缓存。通过 dout@w.T 得到输入梯度并恢复原形状；展平输入转置乘 dout 得到 dW，沿样本轴求和得到 db。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


# 逐元素保留正输入、把非正输入置零，输出 shape 不变。缓存输入供反向判断激活区间；该函数没有可训练参数。
def relu_forward(x):
    """逐元素保留正输入、把非正输入置零，输出 shape 不变。缓存输入供反向判断激活区间；该函数没有可训练参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。

    返回表达式：(out, cache)。
    """
    out = np.maximum(0, x)
    cache = x
    return out, cache


# 将初始版本梯度与前向 x>0 掩码相乘。负区间导数为零；零点采用实现约定，不用于证明处处可微。
def relu_backward(dout, cache):
    """将初始版本梯度与前向 x>0 掩码相乘。负区间导数为零；零点采用实现约定，不用于证明处处可微。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：dx。
    """
    x = cache
    dx = np.where(x > 0, dout, 0)
    return dx


# 输入 (N,D) 特征与 D 维 gamma/beta。训练时按列计算均值、方差并更新滑动统计，测试时只使用滑动统计；输出与输入同形状，cache 用于手写反向。
def batchnorm_forward(x, gamma, beta, bn_param):
    """输入 (N,D) 特征与 D 维 gamma/beta。训练时按列计算均值、方差并更新滑动统计，测试时只使用滑动统计；输出与输入同形状，cache 用于手写反向。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - gamma：可训练的逐特征缩放系数。
    - beta：可训练的逐特征偏移量。
    - bn_param：模式、数值稳定常数和滑动统计等配置。

    返回表达式：(out, cache)。
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        # 计算输出
        mu = x.mean(axis=0)
        xc = x - mu
        var = np.mean(xc ** 2, axis=0)
        std = np.sqrt(var + eps)
        xn = xc / std
        out = gamma * xn + beta

        cache = (mode, x, gamma, xc, std, xn, out)

        # 更新平均值运行中
        running_mean *= momentum
        running_mean += (1 - momentum) * mu

        # 更新运行中的差异平均值
        running_var *= momentum
        running_var += (1 - momentum) * var
    elif mode == 'test':
        # 使用运行的平均值和差异实现正常化
        std = np.sqrt(running_var + eps)
        xn = (x - running_mean) / std
        out = gamma * xn + beta
        cache = (mode, x, xn, gamma, beta, std)
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # 存储更新的运行意味着返回到 bn_ param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


# 从 dout 依次传播到缩放、标准化、方差和均值分支。共享统计使同一列的所有样本相互关联；返回 dx(N,D)、dgamma(D,) 和 dbeta(D,)。
def batchnorm_backward(dout, cache):
    """从 dout 依次传播到缩放、标准化、方差和均值分支。共享统计使同一列的所有样本相互关联；返回 dx(N,D)、dgamma(D,) 和 dbeta(D,)。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dgamma, dbeta)。
    """
    mode = cache[0]
    if mode == 'train':
        mode, x, gamma, xc, std, xn, out = cache

        N = x.shape[0]
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dxc = dxn / std
        dstd = -np.sum((dxn * xc) / (std * std), axis=0)
        dvar = 0.5 * dstd / std
        dxc += (2.0 / N) * xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / N
    elif mode == 'test':
        mode, x, xn, gamma, beta, std = cache
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(xn * dout, axis=0)
        dxn = gamma * dout
        dx = dxn / std
    else:
        raise ValueError(mode)

    return dx, dgamma, dbeta


# 把 NCHW 转为 (N·H·W,C)，复用二维 BN 对每个通道归一化，再转回 NCHW。gamma/beta 长度是 C，而不是 H·W。
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """把 NCHW 转为 (N·H·W,C)，复用二维 BN 对每个通道归一化，再转回 NCHW。gamma/beta 长度是 C，而不是 H·W。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - gamma：可训练的逐特征缩放系数。
    - beta：可训练的逐特征偏移量。
    - bn_param：模式、数值稳定常数和滑动统计等配置。

    返回表达式：(out, cache)。
    """
    N, C, H, W = x.shape
    x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)
    out = out_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return out, cache


# 先把初始版本 NCHW 梯度转换为二维 BN 布局，执行反向，再恢复原图像布局。输出 dx 和按通道聚合的 dgamma、dbeta。
def spatial_batchnorm_backward(dout, cache):
    """先把初始版本 NCHW 梯度转换为二维 BN 布局，执行反向，再恢复原图像布局。输出 dx 和按通道聚合的 dgamma、dbeta。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dgamma, dbeta)。
    """
    N, C, H, W = dout.shape
    dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
    dx_flat, dgamma, dbeta = batchnorm_backward(dout_flat, cache)
    dx = dx_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
    return dx, dgamma, dbeta


# 输入得分 (N,C) 与标签 (N,)。错误类别 margin=max(0,s_j−s_y+1)，正确类别清零；返回平均损失及对得分的梯度。
def svm_loss(x, y):
    """输入得分 (N,C) 与标签 (N,)。错误类别 margin=max(0,s_j−s_y+1)，正确类别清零；返回平均损失及对得分的梯度。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - y：每个样本对应的整数类别标签。

    返回表达式：(loss, dx)。
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos  # 同一行中，每个违反间隔的错误类别都给真类贡献一个负单位，因此扣除整行的激活数。
    dx /= N
    return loss, dx


# 先逐行减最大 logit，再计算 log-sum-exp 与正确类别交叉熵。得分梯度为 (概率−one_hot)/N，输出标量损失和 (N,C) 梯度。
def softmax_loss(x, y):
    """先逐行减最大 logit，再计算 log-sum-exp 与正确类别交叉熵。得分梯度为 (概率−one_hot)/N，输出标量损失和 (N,C) 梯度。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - y：每个样本对应的整数类别标签。

    返回表达式：(loss, dx)。
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

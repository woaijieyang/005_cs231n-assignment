from builtins import range
import numpy as np


# 输入 x 的首维是样本数 N，其余维展平为 D；计算 (N,D)@(D,M)+(M,) 得到 (N,M)。缓存原 x、w、b，供反向恢复输入形状。
def affine_forward(x, w, b):
    """输入 x 的首维是样本数 N，其余维展平为 D；计算 (N,D)@(D,M)+(M,) 得到 (N,M)。缓存原 x、w、b，供反向恢复输入形状。

    参数在本函数中的角色：
    - x：(N,d₁,…,dₖ) 批次；除首轴外展平为 D=d₁×…×dₖ。
    - w：(D,M) 权重，将每样本 D 个输入变为 M 个输出。
    - b：(M,) 偏置，沿 N 个样本广播。

    返回表达式：(out, cache)。
    """
    out = None
    out = x.reshape(x.shape[0], -1).dot(w) + b
    cache = (x, w, b)
    return out, cache


# 全连接W·X
# 输入为 dout(N,M) 和前向缓存。通过 dout@w.T 得到输入梯度并恢复原形状；展平输入转置乘 dout 得到 dW，沿样本轴求和得到 db。
def affine_backward(dout, cache):
    """输入为 dout(N,M) 和前向缓存。通过 dout@w.T 得到输入梯度并恢复原形状；展平输入转置乘 dout 得到 dW，沿样本轴求和得到 db。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    # 反向传播求梯度
    dx = dout.dot(w.T).reshape(x.shape)
    dw = x.reshape(x.shape[0], -1).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db


# 激活函数
# 逐元素保留正输入、把非正输入置零，输出 shape 不变。缓存输入供反向判断激活区间；该函数没有可训练参数。
def relu_forward(x):
    """逐元素保留正输入、把非正输入置零，输出 shape 不变。缓存输入供反向判断激活区间；该函数没有可训练参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。

    返回表达式：(out, cache)。
    """
    out = None
    # 使用简单的maxium即可
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
    dx, x = None, cache
    # 仅有大于0的有贡献，其他为0
    dx = (x > 0) * dout
    return dx


# 批量正则化
# 输入 (N,D) 特征与 D 维 gamma/beta。训练时按列计算均值、方差并更新滑动统计，测试时只使用滑动统计；输出与输入同形状，cache 用于手写反向。
def batchnorm_forward(x, gamma, beta, bn_param):
    """输入 (N,D) 特征与 D 维 gamma/beta。训练时按列计算均值、方差并更新滑动统计，测试时只使用滑动统计；输出与输入同形状，cache 用于手写反向。

    参数在本函数中的角色：
    - x：(N,D) 批次特征，沿 N 轴分别估计 D 个均值和方差。
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
        sample_mean = np.mean(x, axis=0)  # 只沿样本轴统计，同一特征列共享均值。若误用 axis=1，就会变成按样本归一化。
        sample_var = np.var(x, axis=0)
        xhat = (x - sample_mean) / np.sqrt(sample_var + eps)  # 减去批次均值并按标准差缩放；eps 防止某列在当前批次内为常数时除零。
        out = gamma * xhat + beta
        cache = (gamma, x, sample_mean, sample_var, eps, xhat)
        # 基于momentum的指数衰减，从而估计出运行测试集时的均值和方差：
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean  # 滑动均值混合旧统计与本批均值，为推理提供稳定参照；这里的 momentum 与优化器速度无关。
        running_var = momentum * running_var + (1 - momentum) * sample_var  # 同样更新推理所用的方差。训练输出仍由当前 batch 的 sample_var 计算。
    elif mode == 'test':
        a = gamma / np.sqrt(running_var + eps)
        out = a * x + beta - a * running_mean
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

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
    dx, dgamma, dbeta = None, None, None
    gamma, x, sample_mean, sample_var, eps, x_hat = cache
    N, D = dout.shape
    # 使用反向传播
    dbeta = np.sum(dout, axis=0)

    dgammax = dout  # 便于理解
    dgamma = np.sum(dgammax * x_hat, axis=0)

    dxhat = dgammax * gamma  # 经过 y=γx̂+β 的乘法节点，把传入的 dy 乘 γ，得到对标准化特征的导数。
    xmu = x - sample_mean
    divar = np.sum(dxhat * xmu, axis=0)
    ivar = 1 / (np.sqrt(sample_var + eps))
    dxmu1 = dxhat * ivar
    dsqrtvar = -1. / ((np.sqrt(sample_var + eps)) ** 2) * divar
    dvar = 0.5 * 1. / np.sqrt(sample_var + eps) * dsqrtvar
    dsq = 1. / N * np.ones((N, D)) * dvar
    dxmu2 = 2 * xmu * dsq
    dx1 = (dxmu1 + dxmu2)
    dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
    dx2 = 1. / N * np.ones((N, D)) * dmu
    dx = dx1 + dx2

    return dx, dgamma, dbeta


# 将 BN 链式求导代数化简为按列求和和逐元素操作。输入输出与普通反向相同，须先数值比较，再判断性能收益。
def batchnorm_backward_alt(dout, cache):
    """将 BN 链式求导代数化简为按列求和和逐元素操作。输入输出与普通反向相同，须先数值比较，再判断性能收益。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dgamma, dbeta)。
    """
    dx, dgamma, dbeta = None, None, None
    gamma, x, mean, var, eps, xhat = cache
    N = x.shape[0]
    # 直接按照公式计算
    a = np.sqrt(var + eps)
    dxhat = dout * gamma
    dvar = np.sum((x - mean) * dxhat * -0.5 / a ** 3, axis=0)
    dmean = np.sum(- dxhat / a, axis=0) + dvar * np.sum(-2 * (x - mean), axis=0)
    dx = dxhat / a + dvar * 2 * (x - mean) / N + dmean / N

    dgamma = np.sum(dout * xhat, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dx, dgamma, dbeta


# 训练时生成保留概率 p 的掩码，并除以 p 保持期望幅值；测试时直接返回输入。缓存实际掩码供反向复用，本仓库 p 表示保留而非丢弃概率。
def dropout_forward(x, dropout_param):
    """训练时生成保留概率 p 的掩码，并除以 p 保持期望幅值；测试时直接返回输入。缓存实际掩码供反向复用，本仓库 p 表示保留而非丢弃概率。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - dropout_param：保留概率、模式及可选随机种子。

    返回表达式：(out, cache)。
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        mask = (np.random.rand(x.shape[0], x.shape[1]) < p) / p  # 比较结果是保留指示量；除以 p 使训练输出的期望与推理输出一致。这个掩码还要保存给反向传播。
        out = x * mask
    elif mode == 'test':
        out = x

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


# 训练模式把 dout 与前向生成的同一掩码相乘，测试模式直接传递 dout。不能在反向时重新抽样，否则梯度不对应刚执行的前向。
def dropout_backward(dout, cache):
    """训练模式把 dout 与前向生成的同一掩码相乘，测试模式直接传递 dout。不能在反向时重新抽样，否则梯度不对应刚执行的前向。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：dx。
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        # 反向传播，为0项不参与贡献
        dx = dout * mask  # 反向必须使用前向保存的同一掩码。被屏蔽的位置获得零梯度，保留位置按同一比例缩放。
    elif mode == 'test':
        dx = dout
    return dx


# 输入 x(N,C,H,W)、卷积核 w(F,C,HH,WW) 和 F 维偏置。先 padding，再按 stride 滑动窗口，逐窗口对通道和核空间求和；输出为 N×F×Hout×Wout。
def conv_forward_naive(x, w, b, conv_param):
    """输入 x(N,C,H,W)、卷积核 w(F,C,HH,WW) 和 F 维偏置。先 padding，再按 stride 滑动窗口，逐窗口对通道和核空间求和；输出为 N×F×Hout×Wout。

    参数在本函数中的角色：
    - x：(N,C,H,W) 输入图像。
    - w：(F,C,HH,WW) 卷积核，F 决定输出通道数。
    - b：(F,) 每个输出通道共享一个偏置。
    - conv_param：卷积窗口步幅与边界填充。

    返回表达式：(out, cache)。
    """
    out = None
    stride = conv_param['stride']
    pad = conv_param['pad']
    # F是卷积核数目（即输出的深度），C是通道数（如RGB三通道）
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    # 输出形状
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)
    out = np.zeros((N, F, H_out, W_out))
    # 零填充
    x_pad = np.pad(x,
                   ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   mode='constant',
                   constant_values=0)
    for j in range(H_out):
        for k in range(W_out):
            # 按照stride步长依次选择X_pad中卷积核大小的区域
            x_mask = x_pad[:, :, j * stride:j * stride + HH, k * stride:k * stride + WW]
            for i in range(F):
                # 进行卷积操作
                out[:, i, j, k] = np.sum(x_mask * w[i], axis=(1, 2, 3))
    # 扩展维度，使b变成4维数据和out对应
    out += b[None, :, None, None]
    cache = (x, w, b, conv_param)
    return out, cache


# 依据前向窗口位置把 dout 分别累加到 dx、dw 和 db。窗口重叠时必须使用加法累积；结束后去掉输入梯度的 padding，恢复原输入尺寸。
def conv_backward_naive(dout, cache):
    """依据前向窗口位置把 dout 分别累加到 dx、dw 和 db。窗口重叠时必须使用加法累积；结束后去掉输入梯度的 padding，恢复原输入尺寸。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    pad = conv_param['pad']
    _, _, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape
    # 零填充
    x_pad = np.pad(x,
                   ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   mode='constant',
                   constant_values=0)
    dx = np.zeros_like(x)
    dx_pad = np.zeros_like(x_pad)
    dw = np.zeros_like(w)
    # db保证和形状一致
    db = np.sum(dout, axis=(0, 2, 3))

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
            # dw为dout和x_mask的卷积
            for k in range(F):
                # 固定i,j,k后dout输出的是N数目的一维向量，需要拓展维度使其和x_mask对应
                dw[k, :, :, :] += np.sum((dout[:, k, i, j])[:, None, None, None] * x_mask, axis=0)
            # dx_pad为dout和w的卷积
            for n in range(N):
                # 固定i,j,n后dout输出的是F数目的一维向量，需要拓展维度使其和w对应
                dx_pad[n, :, i * stride:i * stride + HH, j * stride:j * stride + WW] += np.sum(
                    (dout[n, :, i, j])[:, None, None, None] * w, axis=0)
    # 需要舍弃填充的部分
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    return dx, dw, db


# 按每个样本和通道遍历空间窗口，输出窗口最大值。池化不混合通道，也没有权重；cache 保留输入和窗口设置。
def max_pool_forward_naive(x, pool_param):
    """按每个样本和通道遍历空间窗口，输出窗口最大值。池化不混合通道，也没有权重；cache 保留输入和窗口设置。

    参数在本函数中的角色：
    - x：(N,C,H,W) 输入，各通道独立池化。
    - pool_param：池化窗口大小与步幅。

    返回表达式：(out, cache)。
    """
    out = None
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H, W = x.shape
    H_out = int(1 + (H - pool_height) / stride)
    W_out = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, i * stride:i * stride + pool_height,
                     j * stride:j * stride + pool_width]
            # 找到（:,:,H,W）也就是axis=(2, 3)最大值
            out[:, :, i, j] = np.max(x_mask, axis=(2, 3))  # 仅沿窗口高度和宽度取最大值，样本维与通道维都保留。
    cache = (x, pool_param)
    return out, cache


# 根据缓存找到窗口最大值的位置，将对应输出梯度回传到输入；当前切片使用赋值，作业采用的不重叠窗口可这样写；重叠窗口在数学上需要累加，而当前函数不支持这个推广。并列最大值按掩码同时接收上游值。
def max_pool_backward_naive(dout, cache):
    """根据缓存找到窗口最大值的位置，将对应输出梯度回传到输入；当前切片使用赋值，作业采用的不重叠窗口可这样写；重叠窗口在数学上需要累加，而当前函数不支持这个推广。并列最大值按掩码同时接收上游值。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：dx。
    """
    dx = None
    x, pool_param = cache
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N, C, H_out, W_out = dout.shape
    dx = np.zeros_like(x)

    for i in range(H_out):
        for j in range(W_out):
            x_mask = x[:, :, i * stride:i * stride + pool_height,
                     j * stride:j * stride + pool_width]
            # 这里找到最大值的位置，最大值矩阵是二维需要拉伸到四维和x_mask保持一致
            mask = (x_mask == np.max(x_mask, axis=(2, 3))[:, :, None, None])
            # 反向只有mask中最大位置有贡献，其余为0
            # 这里是赋值，不是 +=。若步幅小于窗口，后写入的窗口会覆盖此前贡献。
            dx[:, :, i * stride:i * stride + pool_height,
            j * stride:j * stride + pool_width] = \
                dout[:, :, i, j][:, :, None, None] * mask
    return dx


# 把 NCHW 转为 (N·H·W,C)，复用二维 BN 对每个通道归一化，再转回 NCHW。gamma/beta 长度是 C，而不是 H·W。
def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """把 NCHW 转为 (N·H·W,C)，复用二维 BN 对每个通道归一化，再转回 NCHW。gamma/beta 长度是 C，而不是 H·W。

    参数在本函数中的角色：
    - x：(N,C,H,W) 特征图，每个通道在 N·H·W 个位置上共同估计统计量。
    - gamma：可训练的逐特征缩放系数。
    - beta：可训练的逐特征偏移量。
    - bn_param：模式、数值稳定常数和滑动统计等配置。

    返回表达式：(out, cache)。
    """
    out, cache = None, None

    N, C, H, W = x.shape
    # 将(N, C, H, W)变为(N*H*W, C)
    a, cache = batchnorm_forward(x.transpose(0, 2, 3, 1).reshape((N * H * W, C)),
                                 gamma, beta, bn_param)
    # 将(N*H*W, C)变为(N, C, H, W)
    out = a.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return out, cache


# 先把初始版本 NCHW 梯度转换为二维 BN 布局，执行反向，再恢复原图像布局。输出 dx 和按通道聚合的 dgamma、dbeta。
def spatial_batchnorm_backward(dout, cache):
    """先把初始版本 NCHW 梯度转换为二维 BN 布局，执行反向，再恢复原图像布局。输出 dx 和按通道聚合的 dgamma、dbeta。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dgamma, dbeta)。
    """
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    # 将(N, C, H, W)变为(N*H*W, C)
    dbn, dgamma, dbeta = batchnorm_backward(dout.transpose(0, 2, 3, 1).reshape((N * H * W, C)),
                                            cache)
    # 将(N*H*W, C)变为(N, C, H, W)
    dx = dbn.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta


# 输入得分 (N,C) 与标签 (N,)。错误类别 margin=max(0,s_j−s_y+1)，正确类别清零；返回平均损失及对得分的梯度。
def svm_loss(x, y):
    """输入得分 (N,C) 与标签 (N,)。错误类别 margin=max(0,s_j−s_y+1)，正确类别清零；返回平均损失及对得分的梯度。

    参数在本函数中的角色：
    - x：(N,C) 类别得分，不是原始图像；此函数返回对得分的梯度。
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
    - x：(N,C) logits；按行做 log-softmax 并求对 logits 的梯度。
    - y：每个样本对应的整数类别标签。

    返回表达式：(loss, dx)。
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

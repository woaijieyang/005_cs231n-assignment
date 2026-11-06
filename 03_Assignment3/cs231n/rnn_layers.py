from __future__ import print_function, division
from builtins import range
import numpy as np

"""
此文件定义了经常用于神经元的图层类型
networks.
"""


# 输入 x(N,D)、prev_h(N,H) 及两组权重，计算 next_h=tanh(xWx+prev_hWh+b)。输出隐藏状态和反向缓存，不在单步函数内修改训练参数。
def rnn_step_forward(x, prev_h, Wx, Wh, b):
    """输入 x(N,D)、prev_h(N,H) 及两组权重，计算 next_h=tanh(xWx+prev_hWh+b)。输出隐藏状态和反向缓存，不在单步函数内修改训练参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - prev_h：上一时间步的隐藏状态。
    - Wx：从输入到隐藏状态的权重。
    - Wh：从前一隐藏状态到当前状态的权重。
    - b：与输出维对齐的偏置。

    返回表达式：(next_h, cache)。
    """
    next_h, cache = None, None
    # 前向传播
    # ht=tanh(Wx@Xt+Wh@ht-1+b)
    next_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + b)

    cache = (x, prev_h, Wx, Wh, b, next_h)
    return next_h, cache


# 输入 dnext_h 和单步 cache，先乘 tanh 导数，再分别沿 xWx 与 hWh 回传。返回 dx、dprev_h、dWx、dWh、db，权重梯度在 batch 轴聚合。
def rnn_step_backward(dnext_h, cache):
    """输入 dnext_h 和单步 cache，先乘 tanh 导数，再分别沿 xWx 与 hWh 回传。返回 dx、dprev_h、dWx、dWh、db，权重梯度在 batch 轴聚合。

    参数在本函数中的角色：
    - dnext_h：下一隐藏状态收到的上游梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dprev_h, dWx, dWh, db)。
    """
    dx, dprev_h, dWx, dWh, db = None, None, None, None, None
    x, prev_h, Wx, Wh, b, next_h = cache
    # 反向传播先经过tanh
    dtanh = dnext_h * (1 - next_h ** 2)  # (N,H)
    # 线性项求和即可，注意形状即可
    db = np.sum(dtanh, axis=0)

    dWx = x.T.dot(dtanh)  # (D,N) * (N,H)

    dWh = prev_h.T.dot(dtanh)  # (H,N) * (N,H)

    dprev_h = dtanh.dot(Wh.T)  # (N,H) * (H,H)= (N,H)

    dx = dtanh.dot(Wx.T)  # (N,H)*(H,D)=(N,D)
    return dx, dprev_h, dWx, dWh, db


# 输入完整序列 (N,T,D) 和初始隐藏状态，从 t=0 递推到 T−1。每步共享同一组 Wx、Wh、b，输出所有时刻隐藏状态并按时间保存 cache。
def rnn_forward(x, h0, Wx, Wh, b):
    """输入完整序列 (N,T,D) 和初始隐藏状态，从 t=0 递推到 T−1。每步共享同一组 Wx、Wh、b，输出所有时刻隐藏状态并按时间保存 cache。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - h0：第一个时间步之前的隐藏状态。
    - Wx：从输入到隐藏状态的权重。
    - Wh：从前一隐藏状态到当前状态的权重。
    - b：与输出维对齐的偏置。

    返回表达式：(h, cache)。
    """
    h, cache = None, None
    N, T, D = x.shape
    h = np.zeros((N, T, h0.shape[1]))
    cache = []
    ht = h0
    for t in range(T):
        ht, cache_step = rnn_step_forward(x[:, t, :], ht, Wx, Wh, b)
        h[:, t, :] = ht
        cache.append(cache_step)
    return h, cache


# 从最后时刻往前遍历，把当前输出梯度与未来状态回传梯度相加。各时间步的共享权重梯度累加，最终回传到 h0，并返回完整 dx(N,T,D)。
def rnn_backward(dh, cache):
    """从最后时刻往前遍历，把当前输出梯度与未来状态回传梯度相加。各时间步的共享权重梯度累加，最终回传到 h0，并返回完整 dx(N,T,D)。

    参数在本函数中的角色：
    - dh：所有时间步隐藏状态收到的上游梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dh0, dWx, dWh, db)。
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, T, H = dh.shape
    D = cache[0][0].shape[1]
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dprev_h = np.zeros((N, H))
    dWx = np.zeros((D, H))
    dWh = np.zeros((H, H))
    db = np.zeros((H,))
    #
    for t in reversed(range(T)):
        # 这里的dout包含两部分从上面过来的dh，还有从右边过来的之前算出来的dpre_h(对于当前就是dnext_h)
        dnext_h = dh[:, t, :] + dprev_h  # 同一隐藏状态有两条去路：本时刻输出，以及下一时刻的递推。求导时两路梯度相加。
        dx[:, t, :], dprev_h, dWxt, dWht, dbt = rnn_step_backward(dnext_h, cache.pop())
        # 参数共享累加
        dWx += dWxt  # 同一权重在所有时间步共用，因此每一步算出的权重梯度都要累加。
        dWh += dWht
        db += dbt
        # 迭代到最后即为h0
        dh0 = dprev_h
    return dx, dh0, dWx, dWh, db


# 用整数词索引 x(N,T) 从词向量矩阵 W(V,D) 逐位置查表，输出 (N,T,D)。索引本身不可求导，学习对象是被选中的词向量行。
def word_embedding_forward(x, W):
    """用整数词索引 x(N,T) 从词向量矩阵 W(V,D) 逐位置查表，输出 (N,T,D)。索引本身不可求导，学习对象是被选中的词向量行。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - W：权重矩阵。

    返回表达式：(out, cache)。
    """
    out, cache = None, None
    out = W[x]
    cache = (x, W)
    return out, cache


# 对重复词索引的初始版本梯度累加到同一 dW 行；使用可处理重复索引的累加方式，避免普通索引赋值丢掉重复贡献。
def word_embedding_backward(dout, cache):
    """对重复词索引的初始版本梯度累加到同一 dW 行；使用可处理重复索引的累加方式，避免普通索引赋值丢掉重复贡献。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：dW。
    """
    dW = None
    x, W = cache
    N, T, D = dout.shape
    dW = np.zeros(W.shape)
    # 词号可能重复。每次出现都要把对应的上游梯度加到同一词向量行，普通高级索引赋值会漏掉重复贡献。
    np.add.at(dW, x.flatten(), dout.reshape(-1, D))  # 按照x的下标给出dout
    return dW


# 按正负输入分别计算 sigmoid，避免 exp 在极大负数或正数处溢出。输入输出形状一致，门控输出范围为 (0,1)。
def sigmoid(x):
    """按正负输入分别计算 sigmoid，避免 exp 在极大负数或正数处溢出。输入输出形状一致，门控输出范围为 (0,1)。

    参数在本函数中的角色：
    - x：进入该层的输入特征。

    返回表达式：top / (1 + z)。
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


# 把线性输出分为 i/f/o/g 四块；前三块 sigmoid，候选记忆 tanh。通过 c_next=f*c_prev+i*g 和 h_next=o*tanh(c_next) 更新，缓存门值和状态供反向。
def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b):
    """把线性输出分为 i/f/o/g 四块；前三块 sigmoid，候选记忆 tanh。通过 c_next=f*c_prev+i*g 和 h_next=o*tanh(c_next) 更新，缓存门值和状态供反向。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - prev_h：上一时间步的隐藏状态。
    - prev_c：上一时间步的细胞状态。
    - Wx：从输入到隐藏状态的权重。
    - Wh：从前一隐藏状态到当前状态的权重。
    - b：与输出维对齐的偏置。

    返回表达式：(next_h, next_c, cache)。
    """
    next_h, next_c, cache = None, None, None
    H = prev_h.shape[1]
    z = x.dot(Wx) + prev_h.dot(Wh) + b  # 输入和上一隐藏状态先投影到 4H 维。后面的四个切片必须共用相同的门顺序。
    i = sigmoid(z[:, 0:H])  # 输入门决定写入多少候选记忆，取值在 0 和 1 之间。
    f = sigmoid(z[:, H:2 * H])  # 遗忘门逐维控制上一细胞状态保留多少。
    o = sigmoid(z[:, 2 * H:3 * H])  # 第三块是输出门；这里的顺序与框架 LSTM 不同，转移权重时要交换对应块。
    g = np.tanh(z[:, 3 * H:4 * H])  # 最后一块用 tanh 产生可正可负的候选内容，不能改用 sigmoid。
    next_c = f * prev_c + i * g  # 旧记忆经 f 保留，新候选经 i 写入。这里是逐元素相乘，四个张量都按样本和隐藏维对齐。
    next_h = o * np.tanh(next_c)  # 输出门控制暴露给下一层的部分，细胞状态仍单独传给下一时间步。
    cache = (i, f, o, g, next_c, prev_c, prev_h, Wx, Wh, x)

    return next_h, next_c, cache


# 先汇合 h_next 和未来 c_next 对细胞状态的梯度，再分别回传 i/f/o/g。按与前向一致的门顺序拼接 dz，计算输入、前状态和权重梯度。
def lstm_step_backward(dnext_h, dnext_c, cache):
    """先汇合 h_next 和未来 c_next 对细胞状态的梯度，再分别回传 i/f/o/g。按与前向一致的门顺序拼接 dz，计算输入、前状态和权重梯度。

    参数在本函数中的角色：
    - dnext_h：下一隐藏状态收到的上游梯度。
    - dnext_c：下一细胞状态收到的上游梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dprev_h, dprev_c, dWx, dWh, db)。
    """
    dx, dh, dc, dWx, dWh, db = None, None, None, None, None, None
    i, f, o, g, next_c, prev_c, prev_h, Wx, Wh, x = cache
    # o只接收Ct的梯度流
    do = dnext_h * np.tanh(next_c) * o * (1 - o)
    # Ct经过tanh其返回的梯度流记作dtanhc
    dtanh_c = dnext_h * o * (1 - np.tanh(next_c) ** 2)
    # i,f,g，接收dnext_c和dtanh_c两个梯度流
    dnext_out = dnext_c + dtanh_c  # 来自未来细胞状态的梯度与从当前隐藏状态经过 tanh 回来的梯度在这里汇合。
    di = dnext_out * g * i * (1 - i)
    df = dnext_out * prev_c * f * (1 - f)
    dg = dnext_out * i * (1 - g ** 2)
    dprev_c = dnext_out * f
    # 把di,df,do,dg连接起来构成dz
    # 按前向 i/f/o/g 顺序拼回 N×4H，随后才能与 Wx、Wh 的对应列相乘。
    dz = np.hstack((di, df, do, dg))  # (N, 4H)
    # 线性部分Wx * Xt +Wh * ht-1 + b = z
    dWx = x.T.dot(dz)  # (D,N) * (N,4H)
    dWh = prev_h.T.dot(dz)  # (H, N) * (N,4H)
    db = np.sum(dz, axis=0)
    dprev_h = dz.dot(Wh.T)  # (N,4H) * (4H,H) =(N,H)
    dx = dz.dot(Wx.T)  # =(N,4H) * (4H,H) =(N,H)

    return dx, dprev_h, dprev_c, dWx, dWh, db


# 以 h0 和全零 c0 开始逐词更新 LSTM，保存每个时间步的 gate/cache。输出 (N,T,H) 隐藏序列，词表投影在外部统一执行。
def lstm_forward(x, h0, Wx, Wh, b):
    """以 h0 和全零 c0 开始逐词更新 LSTM，保存每个时间步的 gate/cache。输出 (N,T,H) 隐藏序列，词表投影在外部统一执行。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - h0：第一个时间步之前的隐藏状态。
    - Wx：从输入到隐藏状态的权重。
    - Wh：从前一隐藏状态到当前状态的权重。
    - b：与输出维对齐的偏置。

    返回表达式：(h, cache)。
    """
    h, cache = None, None
    N, T, D = x.shape
    H = h0.shape[1]
    prev_h = h0
    h = np.zeros((N, T, H))
    prev_c = np.zeros((N, H))
    cache = []
    for t in range(T):
        next_h, next_c, cache_step = lstm_step_forward(x[:, t, :], prev_h, prev_c, Wx, Wh, b)
        h[:, t, :] = next_h
        prev_h = next_h
        prev_c = next_c
        cache.append(cache_step)

    return h, cache


# 逆序传递隐藏与细胞两条状态梯度；共享 Wx、Wh、b 的导数对时间求和。返回 dx、dh0 和三组参数梯度，不能遗漏跨时间的 dnext_c。
def lstm_backward(dh, cache):
    """逆序传递隐藏与细胞两条状态梯度；共享 Wx、Wh、b 的导数对时间求和。返回 dx、dh0 和三组参数梯度，不能遗漏跨时间的 dnext_c。

    参数在本函数中的角色：
    - dh：所有时间步隐藏状态收到的上游梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dh0, dWx, dWh, db)。
    """
    dx, dh0, dWx, dWh, db = None, None, None, None, None
    N, T, H = dh.shape
    _, D = cache[0][-1].shape
    dprev_h = np.zeros((N, H))
    dx = np.zeros((N, T, D))
    dh0 = np.zeros((N, H))
    dWx = np.zeros((D, 4 * H))
    dWh = np.zeros((H, 4 * H))
    db = np.zeros((4 * H,))
    dnext_c = np.zeros((N, H))
    for t in reversed(range(T)):
        # 和RNN同理，LSTM的dout包含两部分从上面过来的dh，还有从右边过来的之前算出来的dpre_h(对于当前就是dnext_h)
        dnext_h = dh[:, t, :] + dprev_h  # 同一隐藏状态有两条去路：本时刻输出，以及下一时刻的递推。求导时两路梯度相加。
        dx[:, t, :], dprev_h, dprev_c, dWxt, dWht, dbt = lstm_step_backward(dnext_h, dnext_c, cache[t])
        # 每次需要迭代Ct
        dnext_c = dprev_c
        # 参数共享
        dWx += dWxt  # 同一权重在所有时间步共用，因此每一步算出的权重梯度都要累加。
        dWh += dWht
        db += dbt
        # 迭代到最后即为h0
        dh0 = dprev_h

    return dx, dh0, dWx, dWh, db


# 将 (N,T,D) 展平为 (N·T,D) 后乘共享投影矩阵，恢复 (N,T,M) 并加偏置。所有时间步使用同一个词表映射。
def temporal_affine_forward(x, w, b):
    """将 (N,T,D) 展平为 (N·T,D) 后乘共享投影矩阵，恢复 (N,T,M) 并加偏置。所有时间步使用同一个词表映射。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - w：本层的权重。
    - b：与输出维对齐的偏置。

    返回表达式：(out, cache)。
    """
    N, T, D = x.shape
    M = b.shape[0]
    out = x.reshape(N * T, D).dot(w).reshape(N, T, M) + b
    cache = x, w, b, out
    return out, cache


# 把 dout(N,T,M) 展平，对输入与共享权重求导；db 同时沿样本和时间轴求和。输出恢复原序列形状。
def temporal_affine_backward(dout, cache):
    """把 dout(N,T,M) 展平，对输入与共享权重求导；db 同时沿样本和时间轴求和。输出恢复原序列形状。

    参数在本函数中的角色：
    - dout：后续层传回的输出梯度。
    - cache：本层前向为反向计算保留的中间值。

    返回表达式：(dx, dw, db)。
    """
    x, w, b, out = cache
    N, T, D = x.shape
    M = b.shape[0]

    dx = dout.reshape(N * T, M).dot(w.T).reshape(N, T, D)
    dw = dout.reshape(N * T, M).T.dot(x.reshape(N * T, D)).T
    db = dout.sum(axis=(0, 1))

    return dx, dw, db


# 计算每个时间位置的交叉熵，用 mask 屏蔽 NULL 填充，再对 batch 数 N 平均。输出对 logits 的梯度也乘同一 mask，因此填充位置不会更新参数。
def temporal_softmax_loss(x, y, mask, verbose=False):
    """计算每个时间位置的交叉熵，用 mask 屏蔽 NULL 填充，再对 batch 数 N 平均。输出对 logits 的梯度也乘同一 mask，因此填充位置不会更新参数。

    参数在本函数中的角色：
    - x：进入该层的输入特征。
    - y：每个样本对应的整数类别标签。
    - mask：标记哪些位置参与当前运算的掩码。

    返回表达式：(loss, dx)。
    """

    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)  # 和字幕、词表得分使用完全相同的展平顺序，使掩码的每一项对应正确的时间位置。

    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N * T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N * T), y_flat] -= 1  # 每一行对应一个样本的一个时间步。在真实词那一列减一，得到 P−Y。
    dx_flat /= N
    dx_flat *= mask_flat[:, None]  # 掩码从 NT 变为 NT×1，广播到全部词类；同一填充位置的 V 个梯度全部清零。

    if verbose: print('dx_flat: ', dx_flat.shape)

    dx = dx_flat.reshape(N, T, V)

    return loss, dx

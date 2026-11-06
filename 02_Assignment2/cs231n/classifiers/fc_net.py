from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """两层全连接网络：输入先做仿射变换和 ReLU，再映射到类别得分。训练时计算交叉熵与权重正则项，并返回每个参数的梯度。"""

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """按输入维 D、隐藏维 H 和类别数 C 初始化 W1(D,H)、W2(H,C) 与两个零偏置。这个两层类不启用 BN 或 Dropout。

        逐项参数：
        - input_dim，默认 3 * 32 * 32：输入到模型的特征宽度，决定第一层与输入相乘的维度。卷积模型中它是 (C,H,W)。
        - hidden_dim，默认 100：隐藏表示的宽度 H；增加它同时增加计算量和参数量。
        - num_classes，默认 10：分类目标类别数 C，每个样本返回 C 个 logits。
        - weight_scale，默认 0.001：随机权重的初始化尺度；不是正则项系数。
        - reg，默认 0.0：权重 L2 惩罚的系数；课程实现通常使用 ½·reg·ΣW²，导数为 reg·W。
        """
        self.params = {}
        self.reg = reg

        # 使用np.random.normal高斯分布
        self.params['W1'] = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """依次计算 affine→ReLU→affine。没有标签时返回 (N,C) 得分；有标签时加上 Softmax 交叉熵和权重正则项，再逆序返回 W1、b1、W2、b2 的梯度。

        逐项参数：
        - X：按行排列的输入样本，样本轴必须与标签顺序一致。
        - y，默认 None：与样本行一一对应的类别编号，作为监督目标。

        返回给调用方：(loss, grads) 或 scores。
        """
        scores = None
        hidden, affine_relu_cache = affine_relu_forward(X, self.params['W1'],
                                                        self.params['b1'])
        scores, fc_cache = affine_forward(hidden,
                                          self.params['W2'],
                                          self.params['b2'])

        if y is None:
            return scores

        loss, grads = 0, {}
        # 前向传播
        loss, dout = softmax_loss(scores, y)
        # L2正则化损失reg*|W|^2
        loss += 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1']) + \
                0.5 * self.reg * (np.sum(self.params['W2'] * self.params['W2']))

        # 后向传播
        dhidden, dw2, grads['b2'] = affine_backward(dout, fc_cache)
        # 梯度需要加入正则化损失
        grads['W2'] = dw2 + self.reg * self.params['W2']

        dx, dw1, grads['b1'] = affine_relu_backward(dhidden, affine_relu_cache)
        # 梯度需要加入正则化损失
        grads['W1'] = dw1 + self.reg * self.params['W1']


        return loss, grads


class FullyConnectedNet(object):
    """由多个隐藏层组成的分类模型。每个隐藏层执行仿射变换和 ReLU，可按配置加入批量归一化与随机失活。最后一层产生类别得分。loss 在训练时还返回参数梯度。"""

    # 根据层宽列表初始化各层 W/b；启用 BN 时还初始化 gamma/beta 和滑动统计，启用 dropout 时保存保留概率与模式。参数维度由相邻层宽确定。
    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """根据层宽列表初始化各层 W/b；启用 BN 时还初始化 gamma/beta 和滑动统计，启用 dropout 时保存保留概率与模式。参数维度由相邻层宽确定。

        逐项参数：
        - hidden_dims：各隐藏层宽度的有序列表。例如 [100,100] 表示两个隐藏层，分类输出层另算。
        - input_dim，默认 3 * 32 * 32：输入到模型的特征宽度，决定第一层与输入相乘的维度。卷积模型中它是 (C,H,W)。
        - num_classes，默认 10：分类目标类别数 C，每个样本返回 C 个 logits。
        - dropout，默认 0：0 表示关闭随机失活；启用时用作保留概率，而不是丢弃比例。
        - use_batchnorm，默认 False：是否在隐藏层仿射与 ReLU 之间加入 BN；开启时增加 gamma、beta 和滑动统计量。
        - reg，默认 0.0：权重 L2 惩罚的系数；课程实现通常使用 ½·reg·ΣW²，导数为 reg·W。
        - weight_scale，默认 0.01：随机权重的初始化尺度；不是正则项系数。
        - dtype，默认 np.float32：参数数组的数值类型。float64 有利于有限差分，正式训练可用 float32。
        - seed，默认 None：用于可重复的随机抽样，梯度检查中使扰动前后共用同一个掩码。
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        # hidden_dims是隐藏层数量，使用循环建立多层全连接
        for i in range(len(hidden_dims)):
            layer = hidden_dims[i]
            self.params['W' + str(i + 1)] = \
                np.random.normal(0, weight_scale, size=(input_dim, layer))
            self.params['b' + str(i + 1)] = np.zeros(layer)
            if self.use_batchnorm:
                self.params['gamma' + str(i + 1)] = np.ones(layer)
                self.params['beta' + str(i + 1)] = np.zeros(layer)
            input_dim = layer
        self.params['W' + str(self.num_layers)] = \
            np.random.normal(0, weight_scale, size=(input_dim, num_classes))
        self.params['b' + str(self.num_layers)] = np.zeros(num_classes)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    # 按标签是否存在切换 train/test 模式；隐藏层依次 affine、可选 BN、ReLU、可选 dropout，最后输出 logits。训练时计算 Softmax/L2，再逆序通过 cache 返回参数梯度。
    def loss(self, X, y=None):
        """按标签是否存在切换 train/test 模式；隐藏层依次 affine、可选 BN、ReLU、可选 dropout，最后输出 logits。训练时计算 Softmax/L2，再逆序通过 cache 返回参数梯度。

        逐项参数：
        - X：按行排列的输入样本，样本轴必须与标签顺序一致。
        - y，默认 None：与样本行一一对应的类别编号，作为监督目标。

        返回给调用方：(loss, grads) 或 scores。
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        scores = X
        cache = {}
        dropout_cache = {}
        # 循环调用前向和反向传播求梯度和损失
        for i in range(self.num_layers - 1):
            if self.use_batchnorm:
                scores, cache[i] = \
                    affine_bn_relu_forward(scores,
                                           self.params['W' + str(i + 1)],
                                           self.params['b' + str(i + 1)],
                                           self.params['gamma' + str(i + 1)],
                                           self.params['beta' + str(i + 1)],
                                           self.bn_params[i])
            else:
                scores, cache[i] = \
                    affine_relu_forward(scores,
                                        self.params['W' + str(i + 1)],
                                        self.params['b' + str(i + 1)])
            if self.use_dropout:
                scores, dropout_cache[i] = dropout_forward(scores, self.dropout_param)
        # 最后到达输出层的迭代
        scores, cache[self.num_layers - 1] = \
            affine_forward(scores,
                           self.params['W' + str(self.num_layers)],
                           self.params['b' + str(self.num_layers)])

        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        # 从输出层到最后一个隐藏层前反向
        # 求前向求损失
        loss, dout = softmax_loss(scores, y)
        loss += 0.5 * self.reg * np.sum(self.params['W' + str(self.num_layers)] *
                                        self.params['W' + str(self.num_layers)])
        # 求反向梯度
        dhidden, dw, db = affine_backward(dout, cache[self.num_layers - 1])
        grads['W' + str(self.num_layers)] = dw + self.reg * self.params['W' + str(self.num_layers)]
        grads['b' + str(self.num_layers)] = db

        # 隐藏层前反向循环
        # 从num_layers - 1到0，步长为-1
        for i in range(self.num_layers - 1, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params['W' + str(i)] * self.params['W' + str(i)])
            if self.use_dropout:
                dhidden = dropout_backward(dhidden, dropout_cache[i - 1])
            if self.use_batchnorm:
                dhidden, dw, db, dgamma, dbeta = affine_bn_relu_backward(dhidden, cache[i - 1])
                grads['gamma' + str(i)] = dgamma
                grads['beta' + str(i)] = dbeta
            else:
                dhidden, dw, db = affine_relu_backward(dhidden, cache[i - 1])

            grads['W' + str(i)] = dw + self.reg * self.params['W' + str(i)]
            grads['b' + str(i)] = db


        return loss, grads

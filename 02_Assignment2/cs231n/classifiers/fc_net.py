from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """两层全连接网络：输入先做仿射变换和 ReLU，再映射到类别得分。训练时计算交叉熵与权重正则项，并返回每个参数的梯度。"""

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """按输入维 D、隐藏维 H 和类别数 C 初始化 W1(D,H)、W2(H,C) 与两个零偏置。这个两层类不启用 BN 或 Dropout。"""
        self.params = {}
        self.reg = reg

        # 使用np.random.normal高斯分布
        self.params['W1'] = np.random.normal(0, weight_scale, size=(input_dim, hidden_dim))
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = np.random.normal(0, weight_scale, size=(hidden_dim, num_classes))
        self.params['b2'] = np.zeros(num_classes)

    def loss(self, X, y=None):
        """依次计算 affine→ReLU→affine。没有标签时返回 (N,C) 得分；有标签时加上 Softmax 交叉熵和权重正则项，再逆序返回 W1、b1、W2、b2 的梯度。"""
        scores = None
        hidden, affine_relu_cache = affine_relu_forward(X, self.params['W1'],
                                                        self.params['b1'])
        scores, fc_cache = affine_forward(hidden,
                                          self.params['W2'],
                                          self.params['b2'])

        # 如果Y是无,那么我们就在测试模式中了所以只要回分数
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

        参数在本函数中的角色：
        - reg：权重正则化强度。
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

        # 在使用退出时,我们需要通过一个退出_param字典给每一个
        # 退出层, 以便该层了解退出概率和模式
        # (train /test) 。您可以通过同样的退职程序到每个退职层次 。
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # 随着批量正常化,我们需要跟踪运行手段和
        # 差异,所以我们需要通过一个特殊的 bn_ param 对象到每批
        # 常态层。您应该通过 self. bn_ params [0]到前方 。
        # 中,自.bn_params [1]到前端
        # 通过第二批正常化层等.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # 将所有参数投放到正确的数据类型
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    # 按标签是否存在切换 train/test 模式；隐藏层依次 affine、可选 BN、ReLU、可选 dropout，最后输出 logits。训练时计算 Softmax/L2，再逆序通过 cache 返回参数梯度。
    def loss(self, X, y=None):
        """按标签是否存在切换 train/test 模式；隐藏层依次 affine、可选 BN、ReLU、可选 dropout，最后输出 logits。训练时计算 Softmax/L2，再逆序通过 cache 返回参数梯度。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。

        返回表达式：(loss, grads)；scores。
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # 设置列车/测试模式,用于批量停放和退出停放
        # 在训练和测试期间行为不同。
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

        # 如果测试模式提前返回
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

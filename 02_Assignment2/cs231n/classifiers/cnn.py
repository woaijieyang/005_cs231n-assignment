from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """三层卷积分类器。卷积、ReLU 和最大池化提取空间特征，随后经过仿射隐藏层与最终分类层。模型接收 NCHW 图像，返回类别得分或训练损失与梯度。"""

    # 按卷积→ReLU→2×2 池化→仿射→ReLU→仿射建立参数。池化后通道和空间尺寸决定第一全连接权重的行数。
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """按卷积→ReLU→2×2 池化→仿射→ReLU→仿射建立参数。池化后通道和空间尺寸决定第一全连接权重的行数。

        参数在本函数中的角色：
        - reg：权重正则化强度。
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # pad参看loss函数的预设
        pad = (filter_size - 1) // 2
        # 卷积后的输出大小
        H_conv = 1 + (input_dim[1] + 2 * pad - filter_size) / 1
        W_conv = 1 + (input_dim[2] + 2 * pad - filter_size) / 1
        # 池化后的输出大小
        H_pool = 1 + (H_conv - 2) / 2
        W_pool = 1 + (W_conv - 2) / 2
        # 凸轮 - relu - 2x2 最大池
        # W1实现卷积
        self.params['W1'] = np.random.normal(0, weight_scale,
                                             size=(num_filters, input_dim[0],
                                                   filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        # 池化后affine - relu
        # W2实现全连接
        self.params['W2'] = np.random.normal(0, weight_scale,
                                             size=(num_filters * int(H_pool * W_pool),
                                                   hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        # affine - Softmax
        # W3实现评分函数
        self.params['W3'] = np.random.normal(0, weight_scale,
                                             size=(hidden_dim, num_classes))
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    # 依次调用组合层进行前向，无标签时返回 logits；有标签时计算 Softmax 和仅作用于权重的 L2 正则，逆序返回 W1/b1 到 W3/b3 的梯度。
    def loss(self, X, y=None):
        """依次调用组合层进行前向，无标签时返回 logits；有标签时计算 Softmax 和仅作用于权重的 L2 正则，逆序返回 W1/b1 到 W3/b3 的梯度。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。

        返回表达式：(loss, grads)；scores。
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # 向前进方向通过凸起层 。
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # 最大集合层通过球池_参数到前传
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        # 前向传递
        # 凸轮 - relu - 2x2 最大池
        scores, cache1 = conv_relu_pool_forward(X, self.params['W1'],
                                                self.params['b1'],
                                                conv_param,
                                                pool_param)
        # 池化后affine - relu
        scores, cache2 = affine_relu_forward(scores,
                                             self.params['W2'],
                                             self.params['b2'])
        # affine - Softmax
        scores, cache3 = affine_forward(scores,
                                        self.params['W3'],
                                        self.params['b3'])

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dx = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1 ** 2) + np.sum(W2 ** 2) +
                                  np.sum(W3 ** 2))

        dx, dw3, db3 = affine_backward(dx, cache3)
        grads['W3'] = dw3 + self.reg * W3
        grads['b3'] = db3

        dx, dw2, db2 = affine_relu_backward(dx, cache2)
        grads['W2'] = dw2 + self.reg * W2
        grads['b2'] = db2

        dx, dw1, db1 = conv_relu_pool_backward(dx, cache1)
        grads['W1'] = dw1 + self.reg * W1
        grads['b1'] = db1

        return loss, grads

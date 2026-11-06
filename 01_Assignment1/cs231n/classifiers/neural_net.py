import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """两层全连接网络：输入先做仿射变换和 ReLU，再映射到类别得分。训练时计算交叉熵与权重正则项，并返回每个参数的梯度。"""

    # 按 D→H→C 结构初始化两组权重和零偏置，参数保存在独立 params 字典。
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """按 D→H→C 结构初始化两组权重和零偏置，参数保存在独立 params 字典。

        逐项参数：
        - input_size：每个输入样本展平后的宽度 D，决定 W1 有多少行。
        - hidden_size：隐藏层宽度 H，决定 W1 的列数和 W2 的行数。
        - output_size：输出类别数 C，决定最后一层得分宽度。
        - std，默认 0.0001：初始权重正态分布的标准差，控制起始得分尺度。
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 先计算 affine→ReLU→affine；无标签时返回 logits，有标签时计算稳定 Softmax 和 L2 正则，并手写反向返回每个参数的梯度。
    def loss(self, X, y=None, reg=0.0):
        """先计算 affine→ReLU→affine；无标签时返回 logits，有标签时计算稳定 Softmax 和 L2 正则，并手写反向返回每个参数的梯度。

        逐项参数：
        - X：按行排列的输入样本，样本轴必须与标签顺序一致。
        - y，默认 None：与样本行一一对应的类别编号，作为监督目标。
        - reg，默认 0.0：权重 L2 惩罚的系数；课程实现通常使用 ½·reg·ΣW²，导数为 reg·W。

        返回给调用方：(loss, grads) 或 scores。
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        scores = None
        out1 = np.maximum(0, X.dot(W1) + b1)  # (N,D) 乘 (D,H) 得到 (N,H)，加 H 维偏置后保存激活，反向还要用它判断哪些位置通过 ReLU。
        scores = out1.dot(W2) + b2  # (N,H) 乘 (H,C) 产生每个样本的 C 个 logits，尚未归一化为概率。

        if y is None:
            return scores

        loss = None
        # Softmax损失
        # softmax 对逐行平移不敏感；先减最大值，避免大 logits 导致 inf/NaN。
        scores = scores - np.max(scores, axis=1, keepdims=True)
        correct_class_score = scores[np.arange(N), y].reshape(N, 1)  # 每个样本用自己的标签作列索引，得到 N 个真类得分，而不是选择整批相同的一列。
        exp_sum = np.sum(np.exp(scores), axis=1).reshape(N, 1)  # 对每个样本自己的类别轴求和，再保留列维；不能把不同样本的指数混成一个分母。
        loss = np.sum(np.log(exp_sum) - correct_class_score)
        # 取均值
        loss /= N
        # L2正则化
        loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)


        grads = {}
        # 对Softmax求Dw梯度，反向传播
        # Db梯度就是初始版本的梯度sum
        margin2 = np.exp(scores) / exp_sum  # 逐行归一化得到 (N,C) 概率矩阵；接下来只在真类位置减一。
        margin2[np.arange(N), y] += -1
        margin2 /= N  # 交叉熵在 batch 上取平均，得分导数也必须除以同一 N。

        dW2 = out1.T.dot(margin2)  # 把每个隐藏激活对输出误差的贡献沿 batch 聚合，得到 (H,C) 的第二层权重梯度。
        dW2 += reg * W2
        grads['W2'] = dW2
        grads['b2'] = np.sum(margin2, axis=0)

        margin1 = margin2.dot(W2.T)  # 从类别空间把误差投影回隐藏层，此时尚未乘 ReLU 的局部导数。
        margin1[out1 <= 0] = 0  # 第一次前向没有激活的位置，ReLU 的反向通路关闭；梯度不能继续传到对应第一层权重。
        dW1 = X.T.dot(margin1)  # 输入转置为 (D,N)，乘以 (N,H) 得到与 W1 完全同形状的更新量。
        dW1 += reg * W1
        grads['W1'] = dW1
        grads['b1'] = np.sum(margin1, axis=0)

        return loss, grads

    # 循环抽取 mini-batch，用当前学习率执行 SGD；在预定检查点计算训练/验证准确率并衰减学习率。实际代码没有动量状态，不能把它称为动量 SGD。
    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """循环抽取 mini-batch，用当前学习率执行 SGD；在预定检查点计算训练/验证准确率并衰减学习率。实际代码没有动量状态，不能把它称为动量 SGD。

        逐项参数：
        - X：按行排列的输入样本，样本轴必须与标签顺序一致。
        - y：与样本行一一对应的类别编号，作为监督目标。
        - X_val：验证特征，仅用于检查当前参数的泛化表现。
        - y_val：验证标签，与 X_val 同顺序。
        - learning_rate，默认 0.001：一次更新沿负梯度移动的步长；过大可能导致发散，过小会使预算内更新不足。
        - learning_rate_decay，默认 0.95：在训练器的衰减时机乘到当前学习率的因子。
        - reg，默认 1e-05：权重 L2 惩罚的系数；课程实现通常使用 ½·reg·ΣW²，导数为 reg·W。
        - num_iters，默认 100：更新总次数，每次从训练集重新抽一个 batch。
        - batch_size，默认 200：单次更新抽取多少样本；改变它会改变梯度噪声及一轮的更新次数。
        - verbose，默认 False：是否输出进度，不参与损失或梯度计算。
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # SGD负梯度方向更新
            self.params['W1'] -= learning_rate * grads['W1']  # 从权重中减去学习率乘梯度；这是训练步骤，前面的 loss 只计算梯度。
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']


            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    # 执行两层前向并对 logits 取 argmax，不计算损失或梯度。
    def predict(self, X):
        """执行两层前向并对 logits 取 argmax，不计算损失或梯度。

        逐项参数：
        - X：按行排列的输入样本，样本轴必须与标签顺序一致。

        返回给调用方：y_pred。
        """
        y_pred = None

        out1 = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])  # 每行样本由 D 维映射到 H 维；b1 沿样本轴广播，再用 ReLU 截断负值。
        y_pred = np.argmax(out1.dot(self.params['W2']) + self.params['b2'], axis=1)  # 沿类别轴选最大得分的位置，每行只返回一个标签，所以输出为 (N,) 而不是 (N,C)。

        return y_pred

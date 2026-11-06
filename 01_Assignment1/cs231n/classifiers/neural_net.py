import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
    """两层全连接网络：输入先做仿射变换和 ReLU，再映射到类别得分。训练时计算交叉熵与权重正则项，并返回每个参数的梯度。"""

    # 按 D→H→C 结构初始化两组权重和零偏置，参数保存在独立 params 字典。
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """按 D→H→C 结构初始化两组权重和零偏置，参数保存在独立 params 字典。
        """
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    # 先计算 affine→ReLU→affine；无标签时返回 logits，有标签时计算稳定 Softmax 和 L2 正则，并手写反向返回每个参数的梯度。
    def loss(self, X, y=None, reg=0.0):
        """先计算 affine→ReLU→affine；无标签时返回 logits，有标签时计算稳定 Softmax 和 L2 正则，并手写反向返回每个参数的梯度。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。
        - reg：权重正则化强度。

        返回表达式：(loss, grads)；scores。
        """
        # 从参数词典解包变量
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        # 计算前传
        scores = None
        # input->ReLu->Softmax->output
        out1 = np.maximum(0, X.dot(W1) + b1)  # relu, (N, H) (英语).
        scores = out1.dot(W2) + b2  # (N, C)

        # 如果目标没有给出然后跳出去,我们就完了
        if y is None:
            return scores

        # 计算损失
        loss = None
        # 完成前传并计算损失。这应包括
        # W1和W2的数据丢失和L2正则化。
        # 在可变损失中,该损失应为平面。使用软max
        # 分类器损失。这样,你的结果就和我们的相匹配,乘以
        # 正则化损失减少0.5
        # Softmax损失
        # softmax 对逐行平移不敏感；先减最大值，避免大 logits 导致 inf/NaN。
        scores = scores - np.max(scores, axis=1, keepdims=True)
        correct_class_score = scores[np.arange(N), y].reshape(N, 1)
        exp_sum = np.sum(np.exp(scores), axis=1).reshape(N, 1)
        loss = np.sum(np.log(exp_sum) - correct_class_score)
        # 取均值
        loss /= N
        # L2正则化
        loss += 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)


        # 后传:计算梯度
        grads = {}
        # 对Softmax求Dw梯度，反向传播
        # Db梯度就是初始版本的梯度sum
        margin2 = np.exp(scores) / exp_sum
        margin2[np.arange(N), y] += -1
        margin2 /= N  # (N, C)

        dW2 = out1.T.dot(margin2)  # (H ,C)
        dW2 += reg * W2
        grads['W2'] = dW2
        grads['b2'] = np.sum(margin2, axis=0)

        margin1 = margin2.dot(W2.T)  # (N, H)
        margin1[out1 <= 0] = 0  # 第一次前向没有激活的位置，ReLU 的反向通路关闭；梯度不能继续传到对应第一层权重。
        dW1 = X.T.dot(margin1)  # (D, H)
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

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。
        - learning_rate：每一步参数更新的学习率。
        - reg：权重正则化强度。
        - num_iters：更新总次数。
        - batch_size：每次更新使用的样本数。
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # 使用 SGD 优化自建模中的参数
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            # 使用当前小批量计算损失和梯度
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            # SGD负梯度方向更新
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['b2'] -= learning_rate * grads['b2']


            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # 每个时代,检查火车和Val精度和衰变学习率.
            if it % iterations_per_epoch == 0:
                # 检查准确度
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # 学习率下降
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    # 执行两层前向并对 logits 取 argmax，不计算损失或梯度。
    def predict(self, X):
        """执行两层前向并对 logits 取 argmax，不计算损失或梯度。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。

        返回表达式：y_pred。
        """
        y_pred = None

        out1 = np.maximum(0, X.dot(self.params['W1']) + self.params['b1'])  # relu, (N, H) (英语).
        y_pred = np.argmax(out1.dot(self.params['W2']) + self.params['b2'], axis=1)  # (N, C)

        return y_pred

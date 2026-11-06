import numpy as np
from random import shuffle


# 逐样本稳定计算 Softmax 和交叉熵，再累加各类别权重梯度。输出标量 loss 与和 W 同形状的 dW；减最大值不改变概率。
def softmax_loss_naive(W, X, y, reg):
    """逐样本稳定计算 Softmax 和交叉熵，再累加各类别权重梯度。输出标量 loss 与和 W 同形状的 dW；减最大值不改变概率。

    参数在本函数中的角色：
    - W：权重矩阵。
    - X：输入样本矩阵或图像批次。
    - y：每个样本对应的整数类别标签。
    - reg：权重正则化强度。

    返回表达式：(loss, dW)。
    """
    # 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        # 计算前进行数学化简
        scores = X[i].dot(W)
        # 对同一样本的所有类别减去最大得分，概率及梯度不变，避免 exp 溢出。
        scores -= np.max(scores)
        correct_class_score = scores[y[i]]
        exp_sum = np.sum(np.exp(scores))
        loss += np.log(exp_sum) - correct_class_score
        # 对SoftMax的损失函数求导aL/aW
        dW[:, y[i]] += - X[i]
        for j in range(num_classes):
            dW[:, j] += (np.exp(scores[j]) / exp_sum) * X[i]
    # 取均值
    loss /= num_train
    dW /= num_train
    # 在损失中加入正则化。
    # 正则化
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW


# 批量计算 (N,D)@(D,C) logits，按样本稳定归一化，并用 X.T@(P−Y) 一次计算权重梯度。这里采用 0.5·reg·||W||²，因此正则梯度为 reg·W。
def softmax_loss_vectorized(W, X, y, reg):
    """批量计算 (N,D)@(D,C) logits，按样本稳定归一化，并用 X.T@(P−Y) 一次计算权重梯度。这里采用 0.5·reg·||W||²，因此正则梯度为 reg·W。

    参数在本函数中的角色：
    - W：权重矩阵。
    - X：输入样本矩阵或图像批次。
    - y：每个样本对应的整数类别标签。
    - reg：权重正则化强度。

    返回表达式：(loss, dW)。
    """
    # 初始化损失和梯度为零。
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    # 计算前进行数学化简
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
    exp_sum = np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
    loss += np.sum(np.log(exp_sum) - correct_class_score)

    # 对SoftMax的损失函数求导aL/aW
    margin = np.exp(scores) / exp_sum
    margin[np.arange(num_train), y] += -1  # 概率矩阵的真类位置减一，对应交叉熵的得分梯度 P−Y。其他类别仍保留预测概率。
    dW = X.T.dot(margin)  # 把每个样本的特征乘以对应类别系数，再沿样本维相加，结果与权重 W 的形状相同。

    # 取均值
    loss /= num_train
    dW /= num_train
    # 在损失中加入正则化。
    # 正则化
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W

    return loss, dW

import numpy as np
from random import shuffle


# 逐样本逐错误类别计算 hinge margin，正 margin 才贡献损失和权重梯度。正确类别梯度扣去全部激活错误类别贡献，最后按 N 平均并加正则。
def svm_loss_naive(W, X, y, reg):
    """逐样本逐错误类别计算 hinge margin，正 margin 才贡献损失和权重梯度。正确类别梯度扣去全部激活错误类别贡献，最后按 N 平均并加正则。

    参数在本函数中的角色：
    - W：权重矩阵。
    - X：输入样本矩阵或图像批次。
    - y：每个样本对应的整数类别标签。
    - reg：权重正则化强度。

    返回表达式：(loss, dW)。
    """
    dW = np.zeros(W.shape)  # 梯度缓冲与 W 同为 (D,C)，后面累加每个样本对各类别列的贡献。

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # 设置delta = 1
            if margin > 0:
                loss += margin
                # 可以简单推导出max(0，X_i·W_i-X_i·W_yi)的梯度
                dW[:, j] += X[i]
                dW[:, y[i]] += -X[i]

    # SVM损失函数L=1/N*sum(max(0，X_i·W_i-X_i·W_yi))+reg*|W|^2
    # 取平均值
    loss /= num_train
    dW /= num_train
    # 这里乘以0.5是为了Dw求梯度后为0.5*2*reg*W，方便取整
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    return loss, dW


# 一次 X@W 得到全部类别得分，广播正确类别得分形成 margin 矩阵。通过正 margin 指示矩阵和 X.T 相乘得到 dW，正则化系数与循环版本保持一致。
def svm_loss_vectorized(W, X, y, reg):
    """一次 X@W 得到全部类别得分，广播正确类别得分形成 margin 矩阵。通过正 margin 指示矩阵和 X.T 相乘得到 dW，正则化系数与循环版本保持一致。

    参数在本函数中的角色：
    - W：权重矩阵。
    - X：输入样本矩阵或图像批次。
    - y：每个样本对应的整数类别标签。
    - reg：权重正则化强度。

    返回表达式：(loss, dW)。
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # 梯度缓冲与 W 同为 (D,C)，后面累加每个样本对各类别列的贡献。
    num_train = X.shape[0]

    scores = X.dot(W)
    margin = scores - scores[np.arange(num_train), y].reshape(num_train, 1) + 1  # arange 与标签数组逐行配对，选出 N 个正确类别分数；转成 N×1 才能从同一样本的每一类得分中减去它。
    margin[np.arange(num_train), y] = 0.0  # j！=yi 不计算
    margin = (margin > 0) * margin
    loss += margin.sum()
    # 取平均值
    loss /= num_train
    # 正则化
    loss += 0.5 * reg * np.sum(W * W)

    # 向量化的梯度推导，所有j！=yi的元素贡献一次+X_i，所有j=yi的元素献所有max>0个数次-X_i
    margin = (margin > 0) * 1
    row_sum = np.sum(margin, axis=1)
    margin[np.arange(num_train), y] = -row_sum  # 每个错误类别的激活间隔都要从真类扣一次，所以真类的系数是该行正间隔数的负值。
    dW = X.T.dot(margin)  # 把每个样本的特征乘以对应类别系数，再沿样本维相加，结果与权重 W 的形状相同。
    # 取平均值
    dW /= num_train
    # 正则化
    dW += reg * W

    return loss, dW

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *


class LinearClassifier(object):

    def __init__(self):
        self.W = None

    # 从 X、y 抽取 mini-batch，调用子类 loss 得到损失和 dW，使用 SGD 迭代更新。输出损失历史，SVM/Softmax 共用该循环但各自定义损失。
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """从 X、y 抽取 mini-batch，调用子类 loss 得到损失和 dW，使用 SGD 迭代更新。输出损失历史，SVM/Softmax 共用该循环但各自定义损失。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。
        - learning_rate：每一步参数更新的学习率。
        - reg：权重正则化强度。
        - num_iters：更新总次数。
        - batch_size：每次更新使用的样本数。

        返回表达式：loss_history。
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # 假设 Y 的值为 0... K-1, K是类数
        if self.W is None:
            # 懒散初始化 W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # 运行斜度下降以优化W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            # 评估损失和梯度
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # 执行参数更新
            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    # 对每行样本计算 X@W 并取类别轴最大得分，返回一维预测标签。
    def predict(self, X):
        """对每行样本计算 X@W 并取类别轴最大得分，返回一维预测标签。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。

        返回表达式：y_pred。
        """
        y_pred = np.zeros(X.shape[1])
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred

    # 定义具体分类器的损失入口：子类把当前 W、批次样本、标签与正则参数传给对应 SVM 或 Softmax 实现。
    def loss(self, X_batch, y_batch, reg):
        """定义具体分类器的损失入口：子类把当前 W、批次样本、标签与正则参数传给对应 SVM 或 Softmax 实现。

        参数在本函数中的角色：
        - reg：权重正则化强度。
        """
        pass


class LinearSVM(LinearClassifier):
    """线性分类器的 SVM 实现。继承公共 SGD 训练过程，使用多类 hinge 损失计算参数梯度。"""

    # 定义具体分类器的损失入口：子类把当前 W、批次样本、标签与正则参数传给对应 SVM 或 Softmax 实现。
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """线性分类器的 Softmax 实现。继承公共 SGD 训练过程，使用交叉熵计算参数梯度。"""

    # 定义具体分类器的损失入口：子类把当前 W、批次样本、标签与正则参数传给对应 SVM 或 Softmax 实现。
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

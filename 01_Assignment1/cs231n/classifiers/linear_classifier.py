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

        逐项参数：
        - X：按行排列的输入样本，样本轴必须与标签顺序一致。
        - y：与样本行一一对应的类别编号，作为监督目标。
        - learning_rate，默认 0.001：一次更新沿负梯度移动的步长；过大可能导致发散，过小会使预算内更新不足。
        - reg，默认 1e-05：权重 L2 惩罚的系数；课程实现通常使用 ½·reg·ΣW²，导数为 reg·W。
        - num_iters，默认 100：更新总次数，每次从训练集重新抽一个 batch。
        - batch_size，默认 200：单次更新抽取多少样本；改变它会改变梯度噪声及一轮的更新次数。
        - verbose，默认 False：是否输出进度，不参与损失或梯度计算。

        返回给调用方：loss_history。
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # 本题类别编号从 0 开始且连续，因此最大编号加一得到输出列数。
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_classes)

        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            mask = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[mask]
            y_batch = y[mask]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    # 对每行样本计算 X@W 并取类别轴最大得分，返回一维预测标签。
    def predict(self, X):
        """对每行样本计算 X@W 并取类别轴最大得分，返回一维预测标签。

        逐项参数：
        - X：按行排列的输入样本，样本轴必须与标签顺序一致。

        返回给调用方：y_pred。
        """
        y_pred = np.zeros(X.shape[1])
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred

    # 定义具体分类器的损失入口：子类把当前 W、批次样本、标签与正则参数传给对应 SVM 或 Softmax 实现。
    def loss(self, X_batch, y_batch, reg):
        """定义具体分类器的损失入口：子类把当前 W、批次样本、标签与正则参数传给对应 SVM 或 Softmax 实现。

        逐项参数：
        - reg：权重 L2 惩罚的系数；课程实现通常使用 ½·reg·ΣW²，导数为 reg·W。
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

import numpy as np


class KNearestNeighbor(object):
    """保存训练样本与标签，用欧氏距离选择最近的 k 项并投票。"""

    def __init__(self):
        pass

    # 保存 X(N,D) 与 y(N,) 作为最近邻参考样本，不执行优化或参数估计。
    def train(self, X, y):
        """保存 X(N,D) 与 y(N,) 作为最近邻参考样本，不执行优化或参数估计。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。
        """
        self.X_train = X
        self.y_train = y

    # 根据 num_loops 选择距离实现，再调用同一投票逻辑返回查询标签。
    def predict(self, X, k=1, num_loops=0):
        """根据 num_loops 选择距离实现，再调用同一投票逻辑返回查询标签。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - k：参与投票的最近邻数目。
        - num_loops：选择距离实现时使用的显式循环层数。

        返回表达式：self.predict_labels(dists, k=k)。
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    # 两层循环分别遍历查询与训练样本，直接计算欧氏距离。输出 (num_test,num_train)，作为容易核对的基准实现。
    def compute_distances_two_loops(self, X):
        """两层循环分别遍历查询与训练样本，直接计算欧氏距离。输出 (num_test,num_train)，作为容易核对的基准实现。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。

        返回表达式：dists。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.linalg.norm(X[i] - self.X_train[j])
                # 新版本作业不允许使用现成的范数函数，但是实际上范数函数速度会快些
                # dists[i][j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))
        return dists

    # 保留查询样本循环，在全部训练样本上向量化差值与求和。原代码同时写了 norm 和平方和两次计算，第二次覆盖第一次；保留用于理解原始计时。
    def compute_distances_one_loop(self, X):
        """保留查询样本循环，在全部训练样本上向量化差值与求和。原代码同时写了 norm 和平方和两次计算，第二次覆盖第一次；保留用于理解原始计时。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。

        返回表达式：dists。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i, :] = np.linalg.norm(X[i, :] - self.X_train[:], axis=1)
            # 新版本作业不允许使用现成的范数函数,但是实际上范数函数速度会快些
            dists[i, :] = np.sqrt(np.sum(np.square(X[i, :] - self.X_train[:]), axis=1))
        return dists

    # 通过查询范数列、训练范数行与 X@X_train.T 组合平方距离，再开平方。输出完整距离矩阵，内存随查询数与训练数的乘积增长。
    def compute_distances_no_loops(self, X):
        """通过查询范数列、训练范数行与 X@X_train.T 组合平方距离，再开平方。输出完整距离矩阵，内存随查询数与训练数的乘积增长。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。

        返回表达式：dists。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        X_i = (X ** 2).sum(axis=1).reshape((-1, 1))  # 每张查询图沿 3072 个像素特征求平方和。列向量使同一个查询范数能够加到该行的每个训练距离上。
        X_j = (self.X_train ** 2).sum(axis=1).reshape((1, -1))  # 训练图的平方范数排成一行，与查询范数列广播为完整的查询×训练矩阵。
        dists = np.sqrt(X_i + X_j - 2 * X @ self.X_train.T)
        # 同样可以使用numpy自带的矩阵计算函数
        """
        dists = np.sum(np.multiply(X,X)),轴=1, keepdims=True). reshape(num_test, 1).
        dists {np.sum(np.multiply(self. ⁇ train, self. ⁇ train)),轴=1, keepdims=True). reshape(1, num_train).
        dists QQ -2* np.dot(X, self. QQ train.T) 互联网档案馆的存檔,存档日期2014-09-02.
        dists = np.sqrt(dists) 维基百科中的相关条目: 维基语录链接:名人名言 - 分类
        """
        return dists

    # 逐行 argsort 取前 k 个邻居，对训练标签作 bincount 并 argmax。平票时取较小类别编号；该规则在缓存排序方法二中保持一致。
    def predict_labels(self, dists, k=1):
        """逐行 argsort 取前 k 个邻居，对训练标签作 bincount 并 argmax。平票时取较小类别编号；该规则在缓存排序方法二中保持一致。

        参数在本函数中的角色：
        - dists：行是查询样本、列是训练样本的距离矩阵。
        - k：参与投票的最近邻数目。

        返回表达式：y_pred。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # 暂存当前查询样本最近的 k 个训练标签。
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])[:k]]  # 排序得到的是训练行号。先截取距离最近的 k 个行号，再查 y_train 才得到票数统计所需的类别。
            y_pred[i] = np.argmax(np.bincount(closest_y))  # bincount 的第 c 项是类别 c 的票数；argmax 在平票时返回最先出现的最大值，即较小的类别编号。

        return y_pred

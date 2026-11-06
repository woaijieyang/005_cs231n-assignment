import numpy as np


class KNearestNeighbor(object):
    """保存训练样本与标签，用欧氏距离选择最近的 k 项并投票。"""

    def __init__(self):
        pass

    # 保存 X(N,D) 与 y(N,) 作为最近邻参考样本，不执行优化或参数估计。
    def train(self, X, y):
        """保存 X(N,D) 与 y(N,) 作为最近邻参考样本，不执行优化或参数估计。

        逐项参数：
        - X：(M,D) 个参考向量，train 保存这一矩阵；预测函数接收的 X 则是 (N,D) 查询矩阵。
        - y：与样本行一一对应的类别编号，作为监督目标。
        """
        self.X_train = X
        self.y_train = y

    # 根据 num_loops 选择距离实现，再调用同一投票逻辑返回查询标签。
    def predict(self, X, k=1, num_loops=0):
        """根据 num_loops 选择距离实现，再调用同一投票逻辑返回查询标签。

        逐项参数：
        - X：(M,D) 个参考向量，train 保存这一矩阵；预测函数接收的 X 则是 (N,D) 查询矩阵。
        - k，默认 1：每个查询选取距离最小的 k 项投票。需保证参考集非空且 k 合理。
        - num_loops，默认 0：选择双循环、单循环或无循环的距离计算路径，不改变投票规则。

        返回给调用方：self.predict_labels(dists, k=k)。
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

        逐项参数：
        - X：(M,D) 个参考向量，train 保存这一矩阵；预测函数接收的 X 则是 (N,D) 查询矩阵。

        返回给调用方：dists。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = np.linalg.norm(X[i] - self.X_train[j])
                # 新版本作业不允许使用现成的范数函数，但是实际上范数函数速度会快些
        return dists

    # 保留查询样本循环，在全部训练样本上向量化差值与求和。原代码同时写了 norm 和平方和两次计算，第二次覆盖第一次；保留用于理解原始计时。
    def compute_distances_one_loop(self, X):
        """保留查询样本循环，在全部训练样本上向量化差值与求和。原代码同时写了 norm 和平方和两次计算，第二次覆盖第一次；保留用于理解原始计时。

        逐项参数：
        - X：(M,D) 个参考向量，train 保存这一矩阵；预测函数接收的 X 则是 (N,D) 查询矩阵。

        返回给调用方：dists。
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

        逐项参数：
        - X：(M,D) 个参考向量，train 保存这一矩阵；预测函数接收的 X 则是 (N,D) 查询矩阵。

        返回给调用方：dists。
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        X_i = (X ** 2).sum(axis=1).reshape((-1, 1))  # 每张查询图沿 3072 个像素特征求平方和。列向量使同一个查询范数能够加到该行的每个训练距离上。
        X_j = (self.X_train ** 2).sum(axis=1).reshape((1, -1))  # 训练图的平方范数排成一行，与查询范数列广播为完整的查询×训练矩阵。
        dists = np.sqrt(X_i + X_j - 2 * X @ self.X_train.T)  # X_i 是 N×1，X_j 是 1×M，内积是 N×M。三项相加后第 (i,j) 项是平方距离，开方恢复与双循环相同的单位。
        # 同样可以使用numpy自带的矩阵计算函数
        # 距离已由上方的平方范数展开式算出，直接返回完整查询×训练距离矩阵。
        return dists

    # 逐行 argsort 取前 k 个邻居，对训练标签作 bincount 并 argmax。平票时取较小类别编号；该规则在缓存排序方法二中保持一致。
    def predict_labels(self, dists, k=1):
        """逐行 argsort 取前 k 个邻居，对训练标签作 bincount 并 argmax。平票时取较小类别编号；该规则在缓存排序方法二中保持一致。

        逐项参数：
        - dists：(N,M) 距离矩阵，每行查询、每列训练样本；列索引才能用于读取训练标签。
        - k，默认 1：每个查询选取距离最小的 k 项投票。需保证参考集非空且 k 合理。

        返回给调用方：y_pred。
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            closest_y = self.y_train[np.argsort(dists[i])[:k]]  # 排序得到的是训练行号。先截取距离最近的 k 个行号，再查 y_train 才得到票数统计所需的类别。
            y_pred[i] = np.argmax(np.bincount(closest_y))  # bincount 的第 c 项是类别 c 的票数；argmax 在平票时返回最先出现的最大值，即较小的类别编号。

        return y_pred

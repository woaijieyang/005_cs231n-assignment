import numpy as np
from random import shuffle


# 中文数据流：逐样本稳定计算 Softmax 和交叉熵，再累加各类别权重梯度。输出标量 loss 与和 W 同形状的 dW；减最大值不改变概率。
def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
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
    # Add regularization to the loss.
    # 正则化
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


# 中文数据流：批量计算 (N,D)@(D,C) logits，按样本稳定归一化，并用 X.T@(P−Y) 一次计算权重梯度。这里采用 0.5·reg·||W||²，因此正则梯度为 reg·W。
def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # 计算前进行数学化简
    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
    exp_sum = np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
    loss += np.sum(np.log(exp_sum) - correct_class_score)

    # 对SoftMax的损失函数求导aL/aW
    margin = np.exp(scores) / exp_sum
    margin[np.arange(num_train), y] += -1
    dW = X.T.dot(margin)

    # 取均值
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    # 正则化
    loss += 0.5 * reg * np.sum(W * W)
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

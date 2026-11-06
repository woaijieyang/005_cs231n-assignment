from __future__ import print_function, division
from builtins import range
from builtins import object
import numpy as np

from cs231n import optim
from cs231n.coco_utils import sample_coco_minibatch


class CaptioningSolver(object):
    """负责小样本字幕实验。每次抽取字幕及其对应图像特征，调用字幕模型求损失与梯度，再分别更新各参数。大数据 GPU 训练由独立入口执行。"""

    def __init__(self, model, data, **kwargs):
        """接收模型、数据字典与训练超参数，选择更新规则并初始化历史记录。训练集用于更新权重，验证集保留作检查。

        逐项参数：
        - model：提供 loss 接口的模型实例；训练器从它取得同名参数和梯度。
        - data：包含训练与验证划分的字典。分类以图片行采样，字幕以字幕行采样后再找图片特征。
        - update_rule，默认 'sgd'：参数更新函数名，如 sgd、adam；真正执行的规则由这里决定。
        - optim_config，默认 {}：优化器配置，如 learning_rate。训练器为每组参数创建独立状态，不能共享动量缓存。
        - lr_decay，默认 1.0：每轮结束时衰减学习率的倍数。
        - batch_size，默认 100：每次从训练字幕或图像中抽取的行数。
        - num_epochs，默认 10：训练总轮数。
        - print_every，默认 10：每隔多少次更新打印损失。
        - verbose，默认 True：是否显示进度信息。
        """
        self.model = model
        self.data = data

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)

        self.print_every = kwargs.pop('print_every', 10)
        self.verbose = kwargs.pop('verbose', True)

        if len(kwargs) > 0:
            extra = ', '.join('"%s"' % k for k in list(kwargs.keys()))
            raise ValueError('Unrecognized arguments %s' % extra)

        if not hasattr(optim, self.update_rule):
            raise ValueError('Invalid update_rule "%s"' % self.update_rule)
        self.update_rule = getattr(optim, self.update_rule)

        self._reset()

    def _reset(self):
        """初始化轮数、最佳记录、损失和准确率历史，为每个参数分别分配优化器配置。该方法由训练器内部调用。"""
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

        self.optim_configs = {}
        for p in self.model.params:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d

    # 从字幕行抽样，按图像索引取得特征，调用字幕模型的 loss，再逐参数应用优化器。图像 URL 只用于显示，不参与模型输入。
    def _step(self):
        """从字幕行抽样，按图像索引取得特征，调用字幕模型的 loss，再逐参数应用优化器。图像 URL 只用于显示，不参与模型输入。
        """
        minibatch = sample_coco_minibatch(self.data,
                                          batch_size=self.batch_size,
                                          split='train')
        captions, features, urls = minibatch

        loss, grads = self.model.loss(features, captions)
        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    # 初始版本遗留的分类准确率占位接口，直接返回 0.0，训练过程没有调用它。本项目的字幕质量使用独立 BLEU/验证损失计算，不把这里的零值当作实测指标。
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """初始版本遗留的分类准确率占位接口，直接返回 0.0，训练过程没有调用它。本项目的字幕质量使用独立 BLEU/验证损失计算，不把这里的零值当作实测指标。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。
        - batch_size：每次更新使用的样本数。

        返回表达式：0.0；acc。
        """
        return 0.0

        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N / batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.model.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        return acc

    # 按原题轮数执行字幕 mini-batch 更新并记录损失，轮末衰减学习率。该原始 Solver 不做 BLEU 最优模型选择；完整数据 GPU 训练另有独立验证与检查点流程。
    def train(self):
        """按原题轮数执行字幕 mini-batch 更新并记录损失，轮末衰减学习率。该原始 Solver 不做 BLEU 最优模型选择；完整数据 GPU 训练另有独立验证与检查点流程。
        """
        num_train = self.data['train_captions'].shape[0]
        iterations_per_epoch = max(num_train // self.batch_size, 1)
        num_iterations = self.num_epochs * iterations_per_epoch

        for t in range(num_iterations):
            self._step()

            if self.verbose and t % self.print_every == 0:
                print('(Iteration %d / %d) loss: %f' % (
                    t + 1, num_iterations, self.loss_history[-1]))

            epoch_end = (t + 1) % iterations_per_epoch == 0
            if epoch_end:
                self.epoch += 1
                for k in self.optim_configs:
                    self.optim_configs[k]['learning_rate'] *= self.lr_decay

from __future__ import print_function, division
from future import standard_library

standard_library.install_aliases()
from builtins import range
from builtins import object
import os
import pickle as pickle

import numpy as np

from cs231n import optim


class Solver(object):
    """负责分类训练的外层循环。每次从训练集抽一批，调用 model.loss 获得损失和梯度，再用指定更新规则调整参数。每个参数的优化器状态独立保存，验证集用于选择最佳权重。"""

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
        - num_train_samples，默认 1000：计算训练准确率时最多抽取的样本数，不是 SGD 使用的训练集大小。
        - num_val_samples，默认 None：计算验证准确率时最多抽取的验证样本数；None 表示使用整个指定划分。
        - checkpoint_name，默认 None：检查点文件名前缀；None 时不从这个 Solver 写检查点。
        - print_every，默认 10：每隔多少次更新打印损失。
        - verbose，默认 True：是否显示进度信息。
        """
        self.model = model
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']

        self.update_rule = kwargs.pop('update_rule', 'sgd')
        self.optim_config = kwargs.pop('optim_config', {})
        self.lr_decay = kwargs.pop('lr_decay', 1.0)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.num_epochs = kwargs.pop('num_epochs', 10)
        self.num_train_samples = kwargs.pop('num_train_samples', 1000)
        self.num_val_samples = kwargs.pop('num_val_samples', None)

        self.checkpoint_name = kwargs.pop('checkpoint_name', None)
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

    # 抽取一批训练样本，调用 model.loss 获取梯度，再用每组参数独立的优化器状态更新。记录本次 batch 损失。
    def _step(self):
        """抽取一批训练样本，调用 model.loss 获取梯度，再用每组参数独立的优化器状态更新。记录本次 batch 损失。
        """
        num_train = self.X_train.shape[0]
        batch_mask = np.random.choice(num_train, self.batch_size)
        X_batch = self.X_train[batch_mask]
        y_batch = self.y_train[batch_mask]

        loss, grads = self.model.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        for p, w in self.model.params.items():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = self.update_rule(w, dw, config)
            self.model.params[p] = next_w
            self.optim_configs[p] = next_config

    # 将模型及训练配置、epoch、历史指标写到检查点。原函数的检查点并未保存全部优化器缓存，不能声称能够精确恢复所有训练轨迹。
    def _save_checkpoint(self):
        if self.checkpoint_name is None: return
        checkpoint = {
            'model': self.model,
            'update_rule': self.update_rule,
            'lr_decay': self.lr_decay,
            'optim_config': self.optim_config,
            'batch_size': self.batch_size,
            'num_train_samples': self.num_train_samples,
            'num_val_samples': self.num_val_samples,
            'epoch': self.epoch,
            'loss_history': self.loss_history,
            'train_acc_history': self.train_acc_history,
            'val_acc_history': self.val_acc_history,
        }
        filename = '%s_epoch_%d.pkl' % (self.checkpoint_name, self.epoch)
        if self.verbose:
            print('Saving checkpoint to "%s"' % filename)
        with open(filename, 'wb') as f:
            pickle.dump(checkpoint, f)

    # 按批次调用模型测试前向，累计 argmax 预测；可选从给定数据抽样评估。输出正确比例，不执行梯度更新。
    def check_accuracy(self, X, y, num_samples=None, batch_size=100):
        """按批次调用模型测试前向，累计 argmax 预测；可选从给定数据抽样评估。输出正确比例，不执行梯度更新。

        参数在本函数中的角色：
        - X：输入样本矩阵或图像批次。
        - y：每个样本对应的整数类别标签。
        - batch_size：每次更新使用的样本数。

        返回表达式：acc。
        """

        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        num_batches = N // batch_size
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

    # 执行指定轮数的采样更新，周期性评估训练/验证准确率，记录曲线并保留验证最优参数；最终把最佳参数恢复到模型。
    def train(self):

        """执行指定轮数的采样更新，周期性评估训练/验证准确率，记录曲线并保留验证最优参数；最终把最佳参数恢复到模型。
        """

        num_train = self.X_train.shape[0]
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

            first_it = (t == 0)
            last_it = (t == num_iterations - 1)
            if first_it or last_it or epoch_end:
                train_acc = self.check_accuracy(self.X_train, self.y_train,
                                                num_samples=self.num_train_samples)
                val_acc = self.check_accuracy(self.X_val, self.y_val,
                                              num_samples=self.num_val_samples)
                self.train_acc_history.append(train_acc)
                self.val_acc_history.append(val_acc)
                self._save_checkpoint()

                if self.verbose:
                    print('(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                        self.epoch, self.num_epochs, train_acc, val_acc))

                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_params = {}
                    for k, v in self.model.params.items():
                        self.best_params[k] = v.copy()

        self.model.params = self.best_params

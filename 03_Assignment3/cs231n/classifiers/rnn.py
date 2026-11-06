from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.rnn_layers import *


class CaptioningRNN(object):
    """用图像特征预测字幕。投影层把特征映射为初始隐藏状态，词嵌入把整数词号变为向量，循环层按时间处理序列，词表投影给出下一词得分。可选择普通 RNN 或 LSTM。"""

    # 建立图像投影、词嵌入、循环状态和词表投影参数；RNN 循环矩阵宽 H，LSTM 为 4H。dtype 控制参数类型，词表包含 NULL/START/END 的语义索引。
    def __init__(self, word_to_idx, input_dim=512, wordvec_dim=128,
                 hidden_dim=128, cell_type='rnn', dtype=np.float32):
        """建立图像投影、词嵌入、循环状态和词表投影参数；RNN 循环矩阵宽 H，LSTM 为 4H。dtype 控制参数类型，词表包含 NULL/START/END 的语义索引。

        参数在本函数中的角色：
        - word_to_idx：从词字符串映射到整数编号的词表。
        """
        if cell_type not in {'rnn', 'lstm'}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx['<NULL>']
        self._start = word_to_idx.get('<START>', None)
        self._end = word_to_idx.get('<END>', None)

        # 初始化单词矢量
        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        # 初始化 CNN -> 隐藏状态投影参数
        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        # 初始化 RNN 参数
        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        # 初始输出到 vocab 重
        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        # 要校正 dtype 的刻录参数
        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    # 图像特征投影为 h0，captions[:-1] 作为输入词，captions[1:] 为预测目标；循环输出经词表投影计算带 NULL 掩码的损失，再逐层手写反向。
    def loss(self, features, captions):
        """图像特征投影为 h0，captions[:-1] 作为输入词，captions[1:] 为预测目标；循环输出经词表投影计算带 NULL 掩码的损失，再逐层手写反向。

        参数在本函数中的角色：
        - features：按样本排列的视觉特征。
        - captions：一行一个样本的词编号序列。

        返回表达式：(loss, grads)。
        """
        # 将标题切成两块: 标题_ in 包含除最后一个单词以外的所有内容
        # 并将成为 RNN 的输入。 标题_ out 中除了第一个以外, 什么都有
        # 单词,这是我们期待的 rNN 生成的。这些被抵消
        # 由于 RNN应产生单词 (t+1) , 故由单词相对相对 。
        # 收到t 单词后。 首个标题_ in 的元素将是 START
        # 符号,且标题_out的第一个元素将是第一个单词.
        captions_in = captions[:, :-1]  # 每行保留 START 至倒数第二个词，用作教师强制输入。最后一个位置没有下一个目标。
        captions_out = captions[:, 1:]  # 右移一位得到监督目标；第 t 个输出要预测输入第 t 个词之后的词。

        # 你需要这个
        mask = (captions_out != self._null)  # 只对实际词和 END 求损失。PAD 位置既不计入损失也不应更新词表投影。

        # 从图像特性转换到初始特性的重量和偏差
        # 隐藏状态
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        # 文字嵌入矩阵
        W_embed = self.params['W_embed']

        # 输入到隐藏、隐藏到隐藏以及 RNN 的偏差
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        # 重量和偏重 隐藏到伏卡布的转变。
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        # 前向

        # 图片特征 -> FC -> h0
        h0, cache_affine = affine_forward(features, W_proj, b_proj)

        # 单词 -> embedding -> X
        x, cache_embed = word_embedding_forward(captions_in, W_embed)

        # (h0, x) -> RNN -> 输出
        if self.cell_type == 'rnn':
            h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)

        elif self.cell_type == 'lstm':
            h, cache_rnn = lstm_forward(x, h0, Wx, Wh, b)

        # Output -> 时序FC->scores得到每个时序下的得分
        scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)

        # scores-> 时序softmax-> Loss得到每个时序下的损失
        loss, dx = temporal_softmax_loss(scores, captions_out, mask)

        # 反向
        # 时间(_A) bp
        dx, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dx, cache_scores)
        # 韩语bp
        if self.cell_type == 'rnn':
            dx, dh, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dx, cache_rnn)
        elif self.cell_type == 'lstm':
            dx, dh, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dx, cache_rnn)
        # 嵌入的单词 bp
        grads['W_embed'] = word_embedding_backward(dx, cache_embed)
        # affine bp( 双亲)
        dh0, grads['W_proj'], grads['b_proj'] = affine_backward(dh, cache_affine)

        return loss, grads

    # 从图像特征初始化隐藏状态，从 START 开始，逐步嵌入上一预测词、更新 RNN/LSTM 状态并取最大词表得分。输出 token 数组，显示时再截断 END。
    def sample(self, features, max_length=30):
        """从图像特征初始化隐藏状态，从 START 开始，逐步嵌入上一预测词、更新 RNN/LSTM 状态并取最大词表得分。输出 token 数组，显示时再截断 END。

        参数在本函数中的角色：
        - features：按样本排列的视觉特征。

        返回表达式：captions。
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        # 解包参数
        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        # affine 隐藏状态(N,D) -> (N,H)
        h0, _ = affine_forward(features, W_proj, b_proj)

        next_c = np.zeros((N, W_proj.shape[1]))

        word = self._start * np.ones((N,), dtype=np.int32)
        ht = h0
        for i in range(max_length):
            x, _ = word_embedding_forward(word, W_embed)

            if self.cell_type == 'rnn':
                ht, _ = rnn_step_forward(x, ht, Wx, Wh, b)
            else:
                ht, next_c, _ = lstm_step_forward(x, ht, next_c, Wx, Wh, b)

            scores, _ = affine_forward(ht, W_vocab, b_vocab)
            word_predict = scores.argmax(1)
            captions[:, i] = word_predict
            # 自回归解码必须把本步预测作为下一步输入；不能每步都输入 <START>。
            word = word_predict  # 把刚预测的 token 反馈给下一步的 embedding，避免每个时间步都重复读取 START。
        return captions

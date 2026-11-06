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

        逐项参数：
        - word_to_idx：词字符串到整数编号的字典，决定词表宽度及 START、END、NULL 的索引。
        - input_dim，默认 512：视觉特征宽度 D；本题完整训练为 512，决定 W_proj 的行数。
        - wordvec_dim，默认 128：每个词嵌入向量的宽度 W，决定 W_embed 的列数和 Wx 的行数。
        - hidden_dim，默认 128：循环隐藏宽度 H；决定 h、c、W_vocab 的维度，LSTM 的门向量宽为 4H。
        - cell_type，默认 'rnn'：rnn 形成 H 宽的 tanh 变换；lstm 同时计算四个门，仿射宽度为 4H。
        - dtype，默认 np.float32：参数数组的数值类型。float64 有利于有限差分，正式训练可用 float32。
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

        self.params['W_embed'] = np.random.randn(vocab_size, wordvec_dim)
        self.params['W_embed'] /= 100

        self.params['W_proj'] = np.random.randn(input_dim, hidden_dim)
        self.params['W_proj'] /= np.sqrt(input_dim)
        self.params['b_proj'] = np.zeros(hidden_dim)

        dim_mul = {'lstm': 4, 'rnn': 1}[cell_type]
        self.params['Wx'] = np.random.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params['Wx'] /= np.sqrt(wordvec_dim)
        self.params['Wh'] = np.random.randn(hidden_dim, dim_mul * hidden_dim)
        self.params['Wh'] /= np.sqrt(hidden_dim)
        self.params['b'] = np.zeros(dim_mul * hidden_dim)

        self.params['W_vocab'] = np.random.randn(hidden_dim, vocab_size)
        self.params['W_vocab'] /= np.sqrt(hidden_dim)
        self.params['b_vocab'] = np.zeros(vocab_size)

        for k, v in self.params.items():
            self.params[k] = v.astype(self.dtype)

    # 图像特征投影为 h0，captions[:, :-1] 沿时间轴去掉末词作为输入，captions[:, 1:] 去掉起始词作为预测目标；循环输出经词表投影计算带 NULL 掩码的损失，再逐层手写反向。
    def loss(self, features, captions):
        """图像特征投影为 h0，captions[:, :-1] 沿时间轴去掉末词作为输入，captions[:, 1:] 去掉起始词作为预测目标；循环输出经词表投影计算带 NULL 掩码的损失，再逐层手写反向。

        逐项参数：
        - features：(N,D) 图像特征；先通过 W_proj(D,H) 与 b_proj(H,) 形成 h0(N,H)。
        - captions：(N,T+1) 整数字幕。按列切成输入和右移目标，二者都为 (N,T)。

        返回给调用方：(loss, grads)。
        """
        captions_in = captions[:, :-1]  # 每行保留 START 至倒数第二个词，用作教师强制输入。最后一个位置没有下一个目标。
        captions_out = captions[:, 1:]  # 右移一位得到监督目标；第 t 个输出要预测输入第 t 个词之后的词。

        mask = (captions_out != self._null)  # 只对实际词和 END 求损失。PAD 位置既不计入损失也不应更新词表投影。

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']

        W_embed = self.params['W_embed']

        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']

        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

        loss, grads = 0.0, {}
        # 前向

        # 图片特征 -> FC -> h0
        h0, cache_affine = affine_forward(features, W_proj, b_proj)  # (N,D) 图像特征乘 (D,H) 投影矩阵得到 (N,H) 初始状态；cache_affine 留给最后的投影反向。

        # 单词 -> embedding -> X
        x, cache_embed = word_embedding_forward(captions_in, W_embed)  # (N,T) 整数词号查 (V,W) 词表矩阵，输出 (N,T,W) 向量；相同词号引用同一行参数。

        if self.cell_type == 'rnn':
            h, cache_rnn = rnn_forward(x, h0, Wx, Wh, b)  # 从 h0 沿 T 个时间步递推，得到 (N,T,H)；每步缓存按时间顺序保存，反向逆序读取。

        elif self.cell_type == 'lstm':
            h, cache_rnn = lstm_forward(x, h0, Wx, Wh, b)  # LSTM 除 h 外还递推 c，最终同样输出 (N,T,H)，所以后面的词表投影接口可以共用。

        # Output -> 时序FC->scores得到每个时序下的得分
        scores, cache_scores = temporal_affine_forward(h, W_vocab, b_vocab)  # 所有时间步共享 H×V 词表矩阵，将隐藏序列映射成 (N,T,V) 的下一词 logits。

        # scores-> 时序softmax-> Loss得到每个时序下的损失
        loss, dx = temporal_softmax_loss(scores, captions_out, mask)  # dx 此处是对 scores 的梯度；名字虽叫 dx，形状仍为 (N,T,V)。下一行才传回隐藏层。

        # 反向
        dx, grads['W_vocab'], grads['b_vocab'] = temporal_affine_backward(dx, cache_scores)  # dx 在这里改为对隐藏序列的梯度 (N,T,H)。词表权重梯度跨样本和时间两轴累加。
        if self.cell_type == 'rnn':
            dx, dh, grads['Wx'], grads['Wh'], grads['b'] = rnn_backward(dx, cache_rnn)
        elif self.cell_type == 'lstm':
            dx, dh, grads['Wx'], grads['Wh'], grads['b'] = lstm_backward(dx, cache_rnn)
        grads['W_embed'] = word_embedding_backward(dx, cache_embed)  # 按输入 token 将词向量梯度散加回词表。重复词号的贡献不能相互覆盖。
        dh0, grads['W_proj'], grads['b_proj'] = affine_backward(dh, cache_affine)  # dh 是对初始隐藏状态的梯度。投影层反向给出 W_proj 与 b_proj 的梯度；返回到固定图像特征的量不用于训练图像编码器。

        return loss, grads

    # 从图像特征初始化隐藏状态，从 START 开始，逐步嵌入上一预测词、更新 RNN/LSTM 状态并取最大词表得分。输出 token 数组，显示时再截断 END。
    def sample(self, features, max_length=30):
        """从图像特征初始化隐藏状态，从 START 开始，逐步嵌入上一预测词、更新 RNN/LSTM 状态并取最大词表得分。输出 token 数组，显示时再截断 END。

        逐项参数：
        - features：(N,D) 图像特征；先通过 W_proj(D,H) 与 b_proj(H,) 形成 h0(N,H)。
        - max_length，默认 30：最多生成多少个词；这与训练字幕长度和训练轮数是不同概念。

        返回给调用方：captions。
        """
        N = features.shape[0]
        captions = self._null * np.ones((N, max_length), dtype=np.int32)

        W_proj, b_proj = self.params['W_proj'], self.params['b_proj']
        W_embed = self.params['W_embed']
        Wx, Wh, b = self.params['Wx'], self.params['Wh'], self.params['b']
        W_vocab, b_vocab = self.params['W_vocab'], self.params['b_vocab']

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
            captions[:, i] = word_predict  # 将本步最大得分词号写入第 i 列；生成时这一列来自模型预测，而非真实字幕。
            # 自回归解码必须把本步预测作为下一步输入；不能每步都输入 <START>。
            word = word_predict  # 把刚预测的 token 反馈给下一步的 embedding，避免每个时间步都重复读取 START。
        return captions

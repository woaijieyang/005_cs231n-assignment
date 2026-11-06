# CS231n 三份作业：实现、解析与实验

本仓库包含三个 assignment 的 15 个 NumPy/PyTorch notebook、逐块中文解析、实际运行输出和独立改进实验。每份作业与每个 notebook 开头都列出题目要求；方法二直接放在对应方法一下方，保持方法一可独立运行。

## 从哪里开始

第一次读某一题，先看题目要求后面的模型导读。那里有数据流图、公式推导、一个可手算的小例子，以及本题使用各个库的原因。回到代码时，按“第几块｜这一步做什么”的标题阅读，再沿变量名找到产生它的前一块和使用它的后一块。完整汇总里的图和 notebook 使用相同的相对路径。

- [总实现流程](000_总实现流程.md)：按实际操作顺序记录环境、数据、实现、训练、修复及结果。
- [三份作业完整学习记录](002_三份作业完整学习记录.md)：一份 Markdown 包含全部题目、笔记、代码、输出、图像及核心源码附录。文件较长，建议在本地 Markdown 阅读器中打开，并保留 `assets` 目录。
- [全文件索引](001_文件索引.md)：完成作业的文件位置与阅读顺序。
- [问题与解决记录](003_问题与解决记录.md)：实际错误、修复依据、验证和适用边界。
- [Assignment 1](01_Assignment1/000_作业要求.md)：kNN、SVM、Softmax、两层网络、图像特征。
- [Assignment 2](02_Assignment2/000_作业要求.md)：模块化网络、BN、Dropout、卷积、PyTorch。
- [Assignment 3](03_Assignment3/000_作业要求.md)：RNN/LSTM 字幕、网络可视化、风格迁移、GAN。

## 运行环境

Conda 环境、Jupyter 内核注册名和显示名均为 **`cs231n_zx_test`**。实测环境为 Python 3.11、NumPy 1.26.4、PyTorch 2.7.1+cu128、torchvision 0.22.1+cu128，显卡为 RTX 5060 Laptop 8 GB。

```powershell
# 在仓库根目录执行；现有同名环境会先检查，不会删除重建。
.\tools\setup_environment.ps1
conda activate cs231n_zx_test
python tools/run_notebook.py 001
```

已有数据通过 `data_paths.local.json` 或 `CS231N_CIFAR10`、`CS231N_MNIST_GZIP`、`CS231N_COCO` 等环境变量指定。先核验再复用，不按作业或方法复制数据。个人路径配置、数据、权重与检查点不提交 Git；精确依赖在 [requirements-lock.txt](requirements-lock.txt)。

用 `python tools/configure_data.py --help` 查看已有数据路径的配置方式。缺失资源来自课程公开地址：[COCO 特征与字幕](https://cs231n.stanford.edu/coco_captioning.zip)、[ImageNet 验证样例](https://cs231n.stanford.edu/imagenet_val_25.npz)、[SqueezeNet 权重](https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth)。COCO 解压目录与 ImageNet 文件放在项目旁共享 `datasets` 中；SqueezeNet 使用 torch 的共享缓存。已有完整资源不应再次下载。

若只运行方法一，可使用 `python tools/run_notebook.py 013 --original-only`。该选项跳过方法二单元，并且不会覆盖包含全部内容的源 notebook。

## 主要结果

| 实验 | 结果 |
|---|---|
| 两层 NumPy 网络 | 验证 55.2%，测试 54.0% |
| 特征两层网络 | 验证 60.2%，测试 59.2% |
| 模块化全连接网络 | 验证/测试均 52.6% |
| 三层卷积，一轮训练 | 训练 55.8%，验证 53.2% |
| PyTorch 原 VGG 风格模型，10 轮 | 最终整合运行：验证 83.5%，测试 82.17% |
| 独立轻量卷积方法，10 轮 | 最终整合运行：验证 85.1%，测试 82.55%，参数约减少 29.3 倍 |
| 轻量方法增强扩展，总预算 40 轮 | 验证 **90.5%**，测试 **89.75%**，仍为约 115 万参数 |
| kNN 缓存交叉验证 | 逐折结果完全一致，实测约加速 9.4 倍 |
| 完整 COCO RNN/LSTM | 各 20 轮、80,040 次更新；固定验证样本 BLEU-1 为 0.2911 / 0.2817 |
| 冻结字幕模型的束搜索 | 图像不重叠确认子集：RNN 0.2910→0.3033，LSTM 0.2825→0.3094 |
| GAN / LSGAN / DCGAN | 分别完成 10 / 10 / 5 轮并保留生成样本及损失 |

完整 COCO 使用经前向、梯度、解码和优化器对照验证的等价 GPU 后端，原 NumPy 教学实现与小样本训练独立保留。部分方法一参考水平未达到：kNN 固定测试子集为 27.2%，Softmax 验证为 34.6%，原贪心字幕未达到 0.3；束搜索结果另列。两次分类整合运行的准确率存在波动，完整记录见 [重复运行对照](outputs/010_PyTorch/repeatability_comparison.json)，没有挑选较好的一次覆盖其他结果。

## 结果文件与来源

`outputs` 保留精简指标、训练曲线数据、等价性检查及验收记录；模型检查点在本机同名实验目录的 `checkpoints` 下。过程文件留在本机，公开目录只保留完成的作业和结果。

课程题目与接口参考：[CS231n 课程资料](https://cs231n.github.io/)。

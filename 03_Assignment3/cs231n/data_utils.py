from __future__ import print_function

from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
from study_runtime import imread
import platform


# 根据 Python 版本选择 pickle 解码；旧 CIFAR 文件的字符串以 latin1 兼容读取。仅加载明确来源的课程数据，不对任意文件作反序列化。
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


# 读取一个课程 pickle 批次，将扁平像素转成 (10000,32,32,3) HWC 浮点数组，标签转一维整数数组。原文件只读，所有预处理在内存中完成。
def load_CIFAR_batch(filename):
    """读取一个课程 pickle 批次，将扁平像素转成 (10000,32,32,3) HWC 浮点数组，标签转一维整数数组。原文件只读，所有预处理在内存中完成。

    返回表达式：(X, Y)。
    """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y


# 从共享配置解析实际目录，拼接五个训练批次并读取测试批次。输出原始图像/标签，不在这个函数内混合训练与测试统计。
def load_CIFAR10(ROOT):
    from study_runtime import data_path
    ROOT = str(data_path("cifar10"))
    """ 读取完整 CIFAR-10 训练和测试数据。"""
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


# 依次完成划分、用训练均值中心化、HWC→CHW 转换，再把六个数组装入字典。只有训练均值可用于预处理验证与测试数据。
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """依次完成划分、用训练均值中心化、HWC→CHW 转换，再把六个数组装入字典。只有训练均值可用于预处理验证与测试数据。
    """
    # 装入原始 CIFAR- 10 数据
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # 分解数据
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 使数据正常化:减去平均图像
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image  # 训练图的每一行都减去同一训练均值；这里只改变内存中的特征，不修改原始数据文件。
        X_val -= mean_image  # 验证样本必须使用训练阶段估计的均值，否则预处理会使用验证数据自身的信息。
        X_test -= mean_image

    # 切换到频道优先
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # 将数据装入词典
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }


def load_tiny_imagenet(path, dtype=np.float32, subtract_mean=True):
    """读取类别编号、标注和图片，将类别字符串映射为整数索引，把图像转成 NCHW，并按各自划分返回图像与标签。本函数用于 TinyImageNet；本次 SqueezeNet 可视化读取的是 ImageNet 验证样例。"""
    # 第一次负载wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    # 映射到整数标签
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # 使用单词.txt来获取每个类的名称
    with open(os.path.join(path, 'words.txt'), 'r') as f:
        wnid_to_words = dict(line.split('\t') for line in f)
        for wnid, words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]

    # 下一个装入训练数据 。
    X_train = []
    y_train = []
    for i, wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            print('loading training data for synset %d / %d'
                  % (i + 1, len(wnids)))
        # 要找出文件名,我们需要打开框文件
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * \
                        np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                # 灰度文件
                img.shape = (64, 64, 1)
            X_train_block[j] = img.transpose(2, 0, 1)
        X_train.append(X_train_block)
        y_train.append(y_train_block)

    # 我们需要整理所有训练数据
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)

    # 下一个负载验证数据
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # 下次装入测试图像
    # 学生不会有测试标签,所以我们需要删除文件
    # 图像目录。
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]]
                  for img_file in img_files]
        y_test = np.array(y_test)

    mean_image = X_train.mean(axis=0)
    if subtract_mean:
        X_train -= mean_image[None]  # 训练图的每一行都减去同一训练均值；这里只改变内存中的特征，不修改原始数据文件。
        X_val -= mean_image[None]  # 验证样本必须使用训练阶段估计的均值，否则预处理会使用验证数据自身的信息。
        X_test -= mean_image[None]

    return {
        'class_names': class_names,
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_names': class_names,
        'mean_image': mean_image,
    }


def load_models(models_dir):
    """从指定目录逐文件读取保存的模型对象，按文件名建立字典。只应对已知来源的模型文件使用 pickle。"""
    models = {}
    for model_file in os.listdir(models_dir):
        with open(os.path.join(models_dir, model_file), 'rb') as f:
            try:
                models[model_file] = load_pickle(f)['model']
            except pickle.UnpicklingError:
                continue
    return models


def load_imagenet_val(num=None):
    """从指定课程 NPZ 文件读取图像 X、标签 y 以及类别名映射。提供 num 时从前面取指定数量的样本，三者供显著图实验使用。"""
    imagenet_fn = str(__import__('study_runtime').data_path('imagenet_val'))
    if not os.path.isfile(imagenet_fn):
        print('file %s not found' % imagenet_fn)
        print('Run the following:')
        print('cd cs231n/datasets')
        print('bash get_imagenet_val.sh')
        assert False, 'Need to download imagenet_val_25.npz'
    # 课程发布的 label_map 是对象数组；仅对此已核验的课程资源启用 pickle。
    f = np.load(imagenet_fn, allow_pickle=True)
    X = f['X']
    y = f['y']
    class_names = f['label_map'].item()
    if num is not None:
        X = X[:num]
        y = y[:num]
    return X, y, class_names

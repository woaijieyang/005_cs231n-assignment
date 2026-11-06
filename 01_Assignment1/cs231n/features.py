import matplotlib
import numpy as np
from scipy.ndimage import uniform_filter


# 按图像逐一调用给定特征函数，先推导各段长度，再把 HOG/颜色等向量拼到固定列区间。输出 N×总特征维数，不能把不同图像的段顺序打乱。
def extract_features(imgs, feature_fns, verbose=False):
    """按图像逐一调用给定特征函数，先推导各段长度，再把 HOG/颜色等向量拼到固定列区间。输出 N×总特征维数，不能把不同图像的段顺序打乱。

    返回表达式：imgs_features；np.array([])。
    """
    num_images = imgs.shape[0]
    if num_images == 0:
        return np.array([])

    # 使用第一个图像来确定特征尺寸
    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    # 现在我们知道这些特征的维度,我们可以分配一个单一的
    # 大阵列将所有特性作为列存储。
    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

    # 提取其它图像的特性 。
    for i in range(1, num_images):
        idx = 0
        for feature_fn, feature_dim in zip(feature_fns, feature_dims):
            next_idx = idx + feature_dim
            imgs_features[i, idx:next_idx] = feature_fn(imgs[i].squeeze())
            idx = next_idx
        if verbose and i % 1000 == 0:
            print('Done extracting features for %d / %d images' % (i, num_images))

    return imgs_features


# 沿最后 RGB 轴加权求和得到灰度二维图。本仓库保留原权重用于忠实运行验证，HOG 的所有实验采用同一函数，不单独修改测试集变换。
def rgb2gray(rgb):
    """沿最后 RGB 轴加权求和得到灰度二维图。本仓库保留原权重用于忠实运行验证，HOG 的所有实验采用同一函数，不单独修改测试集变换。

    返回表达式：np.dot(rgb[..., :3], [0.299, 0.587, 0.144])。
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])


# 计算水平/垂直像素差形成梯度幅值与方向，按九个方向箱与 8×8 空间单元聚合。32×32 图像输出 4×4×9=144 维方向特征。
def hog_feature(im):
    """计算水平/垂直像素差形成梯度幅值与方向，按九个方向箱与 8×8 空间单元聚合。32×32 图像输出 4×4×9=144 维方向特征。

    返回表达式：orientation_histogram.ravel()。
    """

    # 需要时将 rgb 转换为灰度
    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # 图像大小
    orientations = 9  # 梯度插件数
    cx, cy = (8, 8)  # 每个单元格的像素

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # 在 x 方向上计算梯度
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # 在 Y 方向上计算梯度
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # 梯度
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # 渐变方向

    n_cellsx = int(np.floor(sx / cx))  # x 中的单元格数
    n_cellsy = int(np.floor(sy / cy))  # y 中的单元格数
    # 计算方向集成图像
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        # 为此方向创建新整体图像
        # 在此范围内隔离方向
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
        # 选择这些方向的大小
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, grad_mag, 0)
        orientation_histogram[:, :, i] = uniform_filter(temp_mag, size=(cx, cy))[cx // 2::cx, cy // 2::cy].T

    return orientation_histogram.ravel()


# 先把 RGB 转 HSV，取色相通道按给定箱数统计直方图。返回固定长度颜色分布特征，不保留像素空间位置。
def color_histogram_hsv(im, nbin=10, xmin=0, xmax=255, normalized=True):
    """先把 RGB 转 HSV，取色相通道按给定箱数统计直方图。返回固定长度颜色分布特征，不保留像素空间位置。

    返回表达式：imhist。
    """
    ndim = im.ndim
    bins = np.linspace(xmin, xmax, nbin + 1)
    hsv = matplotlib.colors.rgb_to_hsv(im / xmax) * xmax
    imhist, bin_edges = np.histogram(hsv[:, :, 0], bins=bins, density=normalized)
    imhist = imhist * np.diff(bin_edges)

    # 返回直方图
    return imhist


pass

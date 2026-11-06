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

    feature_dims = []
    first_image_features = []
    for feature_fn in feature_fns:
        feats = feature_fn(imgs[0].squeeze())
        assert len(feats.shape) == 1, 'Feature functions must be one-dimensional'
        feature_dims.append(feats.size)
        first_image_features.append(feats)

    total_feature_dim = sum(feature_dims)
    imgs_features = np.zeros((num_images, total_feature_dim))
    imgs_features[0] = np.hstack(first_image_features).T

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

    if im.ndim == 3:
        image = rgb2gray(im)
    else:
        image = np.at_least_2d(im)

    sx, sy = image.shape  # 灰度图只剩两条空间轴；之后的差分分别沿列方向和行方向计算。
    orientations = 9  # 每个空间单元保留九个方向统计，最后的 HOG 维数为 16×9。
    cx, cy = (8, 8)  # 每个方向直方图汇总一个 8×8 像素单元。

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)  # 比较同一行的相邻列，最右列没有右邻居，保留初始化值。
    gy[:-1, :] = np.diff(image, n=1, axis=0)  # 比较同一列的相邻行，最下行没有下邻居。
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)  # 水平和垂直差分构成二维梯度向量，范数是这个像素给方向箱贡献的权重。
    grad_ori = np.arctan2(gy, (gx + 1e-15)) * (180 / np.pi) + 90  # arctan2 给出方向，乘常数把弧度转成度，再按本实现的偏移映射到后续分箱判断。

    n_cellsx = int(np.floor(sx / cx))  # 沿第一空间轴统计能完整容纳的单元数；本题 32/8 得 4。
    n_cellsy = int(np.floor(sy / cy))  # 沿另一空间轴也得到 4 个单元，总计 16 个空间位置。
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, orientations))
    for i in range(orientations):
        temp_ori = np.where(grad_ori < 180 / orientations * (i + 1),
                            grad_ori, 0)
        temp_ori = np.where(grad_ori >= 180 / orientations * i,
                            temp_ori, 0)
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

    return imhist


pass

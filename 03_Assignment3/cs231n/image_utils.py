from __future__ import print_function
from future import standard_library

standard_library.install_aliases()
from builtins import range
import urllib.request, urllib.error, urllib.parse, os, tempfile

import numpy as np
from study_runtime import imread, imresize

"""
用于查看和处理图像的实用功能.
"""


def blur_image(X):
    """使用固定平滑卷积核对 NCHW 图像作轻微模糊，抑制图像优化时出现的局部高频噪声。"""
    from cs231n.fast_layers import conv_forward_fast
    w_blur = np.zeros((3, 3, 3, 3))
    b_blur = np.zeros(3)
    blur_param = {'stride': 1, 'pad': 1}
    for i in range(3):
        w_blur[i, i] = np.asarray([[1, 2, 1], [2, 188, 2], [1, 2, 1]],
                                  dtype=np.float32)
    w_blur /= 200.0
    return conv_forward_fast(X, w_blur, b_blur, blur_param)[0]


SQUEEZENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
SQUEEZENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img):
    """把像素缩放到 [0,1]，按 SqueezeNet 的通道均值和标准差归一化。"""
    return (img.astype(np.float32) / 255.0 - SQUEEZENET_MEAN) / SQUEEZENET_STD


def deprocess_image(img, rescale=False):
    """撤销通道标准化，必要时调整显示范围，再转为可显示的 uint8 像素。"""
    img = (img * SQUEEZENET_STD + SQUEEZENET_MEAN)
    if rescale:
        vmin, vmax = img.min(), img.max()
        img = (img - vmin) / (vmax - vmin)
    return np.clip(255 * img, 0.0, 255.0).astype(np.uint8)


def image_from_url(url):
    """先按 URL 查找共享缓存中的图片。缺失时下载并解码，验证格式后保存缓存，并返回像素数组。"""
    import hashlib
    import requests
    from study_runtime import data_path
    url = str(url).replace('http://', 'https://', 1)
    cache = data_path('shared') / 'coco_image_cache'
    cache.mkdir(parents=True, exist_ok=True)
    destination = cache / (hashlib.sha256(url.encode()).hexdigest() + '.jpg')
    if destination.exists():
        return imread(destination)
    # 可视化样本按 URL 缓存；不会下载整个 COCO 原图集或重复保存同一图片。
    response = requests.get(url, timeout=(10, 30))
    response.raise_for_status()
    from io import BytesIO
    pixels = imread(BytesIO(response.content))
    destination.write_bytes(response.content)
    return pixels


def load_image(filename, size=None):
    """读取文件并按较短边等比例缩放，返回像素数组。size 为空时保留原尺寸。"""
    img = imread(filename)
    if size is not None:
        orig_shape = np.array(img.shape[:2])
        min_idx = np.argmin(orig_shape)
        scale_factor = float(size) / orig_shape[min_idx]
        new_shape = (orig_shape * scale_factor).astype(int)
        img = imresize(img, scale_factor)
    return img

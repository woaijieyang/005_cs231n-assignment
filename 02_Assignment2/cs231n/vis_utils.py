from builtins import range
from past.builtins import xrange

from math import sqrt, ceil
import numpy as np


# 把多张 HWC 图像按近似方形网格排布，为每张图进行显示缩放并保留间隔。输出是展示像素，不回写模型权重。
def visualize_grid(Xs, ubound=255.0, padding=1):
    """把多张 HWC 图像按近似方形网格排布，为每张图进行显示缩放并保留间隔。输出是展示像素，不回写模型权重。

    返回表达式：grid。
    """
    (N, H, W, C) = Xs.shape
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def vis_grid(Xs):
    """把一批图片缩放并排成网格，返回用于显示的画布。"""
    (N, H, W, C) = Xs.shape
    A = int(ceil(sqrt(N)))
    G = np.ones((A * H + A, A * W + A, C), Xs.dtype)
    G *= np.min(Xs)
    n = 0
    for y in range(A):
        for x in range(A):
            if n < N:
                G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1) * W + x, :] = Xs[n, :, :, :]
                n += 1
    # 常态化为 [0] 1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G


def vis_nn(rows):
    """把分行组织的图片排到同一幅图中，便于逐行比较相似样本。"""
    N = len(rows)
    D = len(rows[0])
    H, W, C = rows[0][0].shape
    Xs = rows[0][0]
    G = np.ones((N * H + N, D * W + D, C), Xs.dtype)
    for y in range(N):
        for x in range(D):
            G[y * H + y:(y + 1) * H + y, x * W + x:(x + 1) * W + x, :] = rows[y][x]
    # 常态化为 [0] 1]
    maxg = G.max()
    ming = G.min()
    G = (G - ming) / (maxg - ming)
    return G

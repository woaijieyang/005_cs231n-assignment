import numpy as np
from random import randrange


# 逐坐标计算中心差分 [f(x+h)−f(x−h)]/(2h)，每次恢复被扰动元素。适用于标量输出；步长过大会跨越非光滑点，过小会产生消去误差。
def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """逐坐标计算中心差分 [f(x+h)−f(x−h)]/(2h)，每次恢复被扰动元素。适用于标量输出；步长过大会跨越非光滑点，过小会产生消去误差。

    参数在本函数中的角色：
    - x：被中心差分逐坐标扰动的数组，可以是模型权重或输入。

    返回表达式：grad。
    """

    fx = f(x)  # 记录未扰动输入处的函数值；后续中心差分另取正、负两侧值。
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # 构造正方向扰动，只改变当前待检坐标。
        fxph = f(x)  # 只在坐标 ix 加上 h 后求值，作为中心差分的正侧。
        x[ix] = oldval - h  # 构造负方向扰动，只改变当前待检坐标。
        fxmh = f(x)  # 只在坐标 ix 减去 h 后求值，作为中心差分的负侧。
        x[ix] = oldval  # 恢复这一项的原值，防止后面检查其他坐标时混入本次扰动。

        grad[ix] = (fxph - fxmh) / (2 * h)  # 两次取值的位置相隔 2h，因此分母是 2h；其余坐标全程固定。
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # 当前坐标已经恢复，现在移动到下一个待检查元素。

    return grad


# 当 f 返回数组时，用初始版本 df 加权输出差分，得到链式法则对应的标量方向导数。每次扰动后恢复 x，返回与 x 同形状的数值梯度。
def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """当 f 返回数组时，用初始版本 df 加权输出差分，得到链式法则对应的标量方向导数。每次扰动后恢复 x，返回与 x 同形状的数值梯度。

    参数在本函数中的角色：
    - x：被加 h、减 h 的输入数组，扰动后必须恢复原值。

    返回表达式：grad。
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h  # 构造正方向扰动，只改变当前待检坐标。
        pos = f(x).copy()
        x[ix] = oldval - h  # 构造负方向扰动，只改变当前待检坐标。
        neg = f(x).copy()
        x[ix] = oldval  # 恢复这一项的原值，防止后面检查其他坐标时混入本次扰动。

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()  # 当前坐标已经恢复，现在移动到下一个待检查元素。
    return grad


def eval_numerical_gradient_blobs(f, inputs, output, h=1e-5):
    """逐坐标扰动输入对象中的数组，两次执行前向计算，将输出变化与上游梯度作内积，得到每个输入的数值梯度。"""
    numeric_diffs = []
    for input_blob in inputs:
        diff = np.zeros_like(input_blob.diffs)
        it = np.nditer(input_blob.vals, flags=['multi_index'],
                       op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            orig = input_blob.vals[idx]

            input_blob.vals[idx] = orig + h
            f(*(inputs + (output,)))
            pos = np.copy(output.vals)
            input_blob.vals[idx] = orig - h
            f(*(inputs + (output,)))
            neg = np.copy(output.vals)
            input_blob.vals[idx] = orig

            diff[idx] = np.sum((pos - neg) * output.diffs) / (2.0 * h)

            it.iternext()  # 当前坐标已经恢复，现在移动到下一个待检查元素。
        numeric_diffs.append(diff)
    return numeric_diffs


def eval_numerical_gradient_net(net, inputs, output, h=1e-5):
    return eval_numerical_gradient_blobs(lambda *args: net.forward(),
                                         inputs, output, h=h)


# 随机挑选少量参数坐标，用中心差分和解析梯度比较。它是快速抽查，不是全部坐标证明；对 hinge/ReLU 折点需结合绝对误差和扰动尺度解释。
def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """随机挑选少量参数坐标，用中心差分和解析梯度比较。它是快速抽查，不是全部坐标证明；对 hinge/ReLU 折点需结合绝对误差和扰动尺度解释。

    参数在本函数中的角色：
    - x：被抽查若干坐标的参数矩阵。
    """

    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # 构造正方向扰动，只改变当前待检坐标。
        fxph = f(x)  # 只在坐标 ix 加上 h 后求值，作为中心差分的正侧。
        x[ix] = oldval - h  # 构造负方向扰动，只改变当前待检坐标。
        fxmh = f(x)  # 只在坐标 ix 减去 h 后求值，作为中心差分的负侧。
        x[ix] = oldval  # 恢复这一项的原值，防止后面检查其他坐标时混入本次扰动。

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' % (grad_numerical, grad_analytic, rel_error))

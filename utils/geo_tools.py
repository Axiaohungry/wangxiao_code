# D:\PycharmProjects\wangxiao_code\utils\geo_tools.py （用于从高斯的旋转参数中提取法线方向）
import torch
import torch.nn.functional as F


def build_rotation(r):
    # 将四元数转换为旋转矩阵
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def compute_normals_from_rotation(rotation):
    """
    3DGS 的高斯球是椭球。
    通常最短的那个轴（Scaling 最小的轴）对应表面的法线方向。
    但在训练不完美时，这个假设不一定全对，我们这里假设 scaling 最小的轴是法线。
    或者更简单的：假设 Z 轴方向分量。

    更稳妥的方式：取旋转矩阵的第三列作为局部 Z 轴方向。
    """
    R = build_rotation(rotation)
    # 取旋转矩阵的第 3 列 (Local Z-axis) 作为法线方向
    # [N, 3]
    normals = R[:, :, 2]
    return normals
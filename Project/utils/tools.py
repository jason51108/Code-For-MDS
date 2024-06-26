import numpy as np


# 距离的平方
def Distance_matrix(Theta):
    diff = Theta[:, np.newaxis, :] - Theta[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    return D

# 计算距离情况下矩阵M(1)
def Distance_matrix_M_1(alpha, Theta):
    n, r = Theta.shape
    ones_n = np.ones((n, 1))

    term1 = np.dot(ones_n, alpha.reshape(1, -1))
    term2 = np.dot(alpha.reshape(-1, 1), ones_n.T)
    D = Distance_matrix(Theta)
    M = term1 + term2 - 0.5 * D
    return M

# 计算距离情况下矩阵M(2)
def Distance_matrix_M_2(alpha, Theta):
    n, r = Theta.shape
    ones_n = np.ones((n, 1))

    term1 = np.dot(ones_n, alpha.reshape(1, -1))
    term2 = np.dot(alpha.reshape(-1, 1), ones_n.T)
    D = Distance_matrix(Theta)
    M = term1 + term2 - D
    return M

# 计算内积情况下矩阵M
def Inner_matrix_M(alpha, Theta):
    n, r = Theta.shape
    ones_n = np.ones((n, 1))

    term1 = np.dot(ones_n, alpha.reshape(1, -1))
    term2 = np.dot(alpha.reshape(-1, 1), ones_n.T)
    D = Theta@Theta.T
    M = term1 + term2 + D
    return M

# 投影函数
def Projection(X, C):
    norm_X = np.linalg.norm(X)
    return X if norm_X < C else C * X / norm_X

# 矩阵列中心化
def Centralize_matrix(matrix):
    column_means = np.mean(matrix, axis=0)
    centralized_matrix = matrix - column_means
    return centralized_matrix

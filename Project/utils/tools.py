import numpy as np
from scipy.special import expit

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


# Binomial产生邻接矩阵(通用)
def Generate_adj_binomial(M):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    adjacency_matrix = np.random.binomial(1, sigmoid_M)
    adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T #保证邻接矩阵是对称的
    return adjacency_matrix

# Binomial对数似然函数(通用)
def Log_likelihood_binomial(A, M):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    result = np.sum(A * M + np.log(1 - sigmoid_M))
    # diagonal_correction = np.sum(np.diag(A * M + np.log(1 - sigmoid(np.diag(M))))) #对角线的似然函数值
    return result

# Binomial对数似然函数(distacne1)
def Gradient_theta_binomial_distacne1(A, M, thetas):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = -(diff * A[:, :, np.newaxis]) + diff * sigmoid_M[:, :, np.newaxis]
    gradient = np.sum(term, axis=1)
    return gradient

# Binomial对数似然函数(distacne2)
def Gradient_theta_binomial_distacne2(A, M, thetas):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = -2*(diff * A[:, :, np.newaxis]) + 2*diff * sigmoid_M[:, :, np.newaxis]
    gradient = np.sum(term, axis=1)
    return gradient

# Binomial对数似然函数(innner-product)
def Gradient_theta_binomial_inner(A, M, thetas):
    sigma_M = expit(M)
    np.fill_diagonal(sigma_M, 0)
    dL_dpi = A - sigma_M
    gradient = np.dot(dL_dpi + dL_dpi.T, thetas)
    return gradient

# Binomial对数似然函数对alpha的导数
def Gradient_alpha_binomial(A, M):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    gradient = np.sum(A - sigmoid_M, axis=1)
    return gradient
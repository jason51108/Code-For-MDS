import numpy as np
from scipy.special import factorial
from scipy.special import expit

# 距离的平方
def Distance_matrix(Theta):
    diff = Theta[:, np.newaxis, :] - Theta[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    return D

# 计算距离情况下矩阵M(1)
def Distance_matrix_M_1(alpha, Theta, rho=0):
    D = Distance_matrix(Theta)
    M = rho - D
    return M

# 计算距离情况下矩阵M(2)
def Distance_matrix_M_2(alpha, Theta, rho=0):
    n, r = Theta.shape
    ones_n = np.ones((n, 1))

    term1 = np.dot(ones_n, alpha.reshape(1, -1))
    term2 = np.dot(alpha.reshape(-1, 1), ones_n.T)
    D = Distance_matrix(Theta)
    M = term1 + term2 - D + rho
    return M

# 计算内积情况下矩阵M
def Inner_matrix_M(alpha, Theta, rho=0):
    n, r = Theta.shape
    ones_n = np.ones((n, 1))

    term1 = np.dot(ones_n, alpha.reshape(1, -1))
    term2 = np.dot(alpha.reshape(-1, 1), ones_n.T)
    D = Theta@Theta.T
    M = term1 + term2 + D + rho
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



"""
The following is the code for the Binomial distribution
"""
# Binomial产生邻接矩阵(通用)
def Generate_adj_binomial(M):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    adjacency_matrix = np.random.binomial(1, sigmoid_M)
    adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T #保证邻接矩阵是对称的
    return adjacency_matrix

# Binomial对数似然函数(未加惩罚项)
def Log_likelihood_binomial(A, M):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    result = np.sum(A * M + np.log(1 - sigmoid_M))
    # diagonal_correction = np.sum(np.diag(A * M + np.log(1 - sigmoid(np.diag(M))))) #对角线的似然函数值
    return result

# Binomial对数似然函数对theta的导数(distacne1)
def Gradient_theta_binomial_distacne1(A, M, thetas):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = -2*(diff * A[:, :, np.newaxis]) + 2*diff * sigmoid_M[:, :, np.newaxis]
    gradient = np.sum(term, axis=1)
    return gradient

# Binomial对数似然函数对theta的导数(distacne2)
def Gradient_theta_binomial_distacne2(A, M, thetas):
    sigmoid_M = expit(M)
    np.fill_diagonal(sigmoid_M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = -2*(diff * A[:, :, np.newaxis]) + 2*diff * sigmoid_M[:, :, np.newaxis]
    gradient = np.sum(term, axis=1)
    return gradient

# Binomial对数似然函数对theta的导数(innner-product)
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



"""
The following is the code for the Poisson distribution
"""
# Poisson产生邻接矩阵(通用)
def Generate_adj_poisson(M):
    exp_M = np.exp(M)
    adjacency_matrix = np.random.poisson(lam = exp_M)
    adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T
    return adjacency_matrix

# Poisson对数似然函数(未加惩罚项)
def Log_likelihood_poisson(A, M):
    exp_M = np.exp(M)
    np.fill_diagonal(exp_M, 0)
    result = np.sum(A*M - exp_M - np.log(factorial(A)))
    # diagonal_correction = np.sum(np.diag(A * M - exp_M - np.log(factorial(A)))) #对角线的似然函数值
    return result

# Poisson对数似然函数对theta的导数(distacne1)
def Gradient_theta_poisson_distacne1(A, M, thetas):
    exp_M = np.exp(M)
    np.fill_diagonal(exp_M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = -2 * (diff * A[:, :, np.newaxis]) + 2 * diff * exp_M[:, :, np.newaxis]
    gradient = np.sum(term, axis=1)
    return gradient

# Poisson对数似然函数对theta的导数(distacne2)
def Gradient_theta_poisson_distacne2(A, M, thetas):
    exp_M = np.exp(M)
    np.fill_diagonal(exp_M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = -2 * (diff * A[:, :, np.newaxis]) + 2 * diff * exp_M[:, :, np.newaxis]
    gradient = np.sum(term, axis=1)
    return gradient

# Poisson对数似然函数对theta的导数(innner-product)
def Gradient_theta_poisson_inner(A, M, thetas):
    pass

# Poisson对数似然函数对alpha的导数
def Gradient_alpha_poisson(A, M):
    exp_M = np.exp(M)
    np.fill_diagonal(exp_M, 0)
    gradient = np.sum(A - exp_M, axis=1)
    return gradient



"""
The following is the code for the Normal distribution
"""
# Normal产生邻接矩阵(通用)
def Generate_adj_normal(M, scale):
    adjacency_matrix = np.random.normal(loc=M, scale=scale)
    adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T
    return adjacency_matrix

# Normal对数似然函数(未加惩罚项)
def Log_likelihood_normal(A, M, scale):
    result = np.sum((A*M - 0.5*M**2)/scale - A**2/(2*scale) - 0.5*np.log(2*np.pi*scale))
    diagonal_correction = np.sum(np.diag((A*M - 0.5*M**2)/scale - A**2/(2*scale) - 0.5*np.log(2*np.pi*scale)))
    return result-diagonal_correction

# Normal对数似然函数对theta的导数(distacne1)
def Gradient_theta_normal_distacne1(A, M, thetas, scale):
    np.fill_diagonal(M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = (-2 * (diff * A[:, :, np.newaxis]) + 2 * diff * M[:, :, np.newaxis])/scale
    gradient = np.sum(term, axis=1)
    return gradient

# Normal对数似然函数对theta的导数(distacne2)
def Gradient_theta_normal_distacne2(A, M, thetas, scale):
    np.fill_diagonal(M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]
    term = (-2 * (diff * A[:, :, np.newaxis]) + 2 * diff * M[:, :, np.newaxis])/scale
    gradient = np.sum(term, axis=1)
    return gradient

# Normal对数似然函数对theta的导数(innner-product)
def Gradient_theta_normal_inner(A, M, thetas, scale):
    pass

# Normal对数似然函数对alpha的导数
def Gradient_alpha_normal(A, M, scale):
    np.fill_diagonal(M, 0)
    gradient = np.sum((A - M)/scale, axis=1)
    return gradient
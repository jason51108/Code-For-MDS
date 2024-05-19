import numpy as np
import matplotlib.pyplot as plt
import random
import time
import statistics
from scipy.special import factorial
import os
import sys

# Normal生成
def generate_adj(alpha, Theta, scale):
    # 计算 m 矩阵
    M = compute_matrix_M(alpha, Theta)

    # # 生成邻接矩阵
    adjacency_matrix = np.random.normal(loc=M, scale=scale)

    # # 由于邻接矩阵应该是对称的，我们取上三角和下三角的最大值
    adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T
    return adjacency_matrix

# fixed
def distance_matrix(Theta):
    diff = Theta[:, np.newaxis, :] - Theta[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    return D

# fixed
def compute_matrix_M(alpha, Theta):
    n, r = Theta.shape
    ones_n = np.ones((n, 1))

    # 计算 1_n * alpha^T 和 alpha * 1_n^T
    term1 = np.dot(ones_n, alpha.reshape(1, -1))
    term2 = np.dot(alpha.reshape(-1, 1), ones_n.T)
    
    # 计算距离矩阵 D 使用向量化方法
    D =  distance_matrix(Theta)

    # 计算最终的矩阵 M
    M = term1 + term2 - D
    return M

# Normal生成
def f(A, alphas, thetas, scale):
    M = compute_matrix_M(alphas, thetas)
    result = np.sum((A*M - 0.5*M**2)/scale - A**2/(2*scale) - 0.5*np.log(2*np.pi*scale))
    diagonal_correction = np.sum(np.diag((A*M - 0.5*M**2)/scale - A**2/(2*scale) - 0.5*np.log(2*np.pi*scale)))
    
    return result-diagonal_correction

def Projection(X,C):
    return X if np.linalg.norm(X) < C else C*X/np.linalg.norm(X)

# Normal生成
def gradient_theta(A, alphas, thetas, sacle):
    # 使用已定义的函数计算 M 矩阵
    M = compute_matrix_M(alphas, thetas)
    
    # 计算每个节点对的 theta 差值
    np.fill_diagonal(M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]  # (num_nodes, num_nodes, num_theta_dims)
    
    # 计算梯度贡献项
    term = (-2 * (diff * A[:, :, np.newaxis]) + 2 * diff * M[:, :, np.newaxis])/sacle
    
    # 求和所有贡献以更新梯度
    gradient = np.sum(term, axis=1)
    
    return gradient

# Normal生成
def gradient_alpha(A, alphas, thetas, scale):
    # 使用之前定义的 compute_matrix_M 函数计算 M
    M = compute_matrix_M(alphas, thetas)
    
    # 注意我们需要移除对角线上的元素，因为 i 不等于 j
    np.fill_diagonal(M, 0)
    
    # 每个 alpha[i] 的梯度由所有 j!=i 的 sigmoid_M[i, j] 贡献
    gradient = np.sum((A - M)/scale, axis=1)
    
    return gradient


def main(num_samples, k, C, initial_learning_rate, tolerance):
    # Generate true parameters
    true_alpha = np.random.rand(num_samples)
    true_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples, -1)

    # Generate adjacency matrix
    adjacency_matrix = generate_adj(true_alpha, true_theta, scale)

    # Generate distance matrix
    # distance_matrix_ = distance_matrix(true_theta)

    # Initialize predicted parameters
    pred_alpha = np.random.rand(num_samples)
    pred_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples, -1)

    # Initial log likelihood
    logli = f(adjacency_matrix, pred_alpha, pred_theta, scale)
    prev_logli = logli  # Initialize previous log likelihood

    # Training loop
    flag = True
    iter = 0
    learning_rate = initial_learning_rate  # Initialize learning rate

    while flag:
        # Update theta
        grad_y = gradient_theta(adjacency_matrix, pred_alpha, pred_theta, scale)
        temp_theta = Projection(pred_theta + learning_rate * grad_y, C)

        # Update alpha
        grad_x = gradient_alpha(adjacency_matrix, pred_alpha, temp_theta, scale)
        temp_alpha = Projection(pred_alpha + learning_rate * grad_x, C)

        # Calculate new log likelihood
        temp_logli = f(adjacency_matrix, temp_alpha, temp_theta, scale)
        
        # Check improvement in log likelihood
        if temp_logli - prev_logli < 0:
            # If log likelihood decreases, reduce learning rate without updating parameters
            learning_rate /= 5
        else:
            # If log likelihood improves, update parameters and reset learning rate
            iter += 1
            pred_theta = temp_theta
            pred_alpha = temp_alpha
            logli = temp_logli

            # Check for convergence
            if logli - prev_logli > 0 and (logli - prev_logli)/np.abs(prev_logli) < tolerance:
                flag = False
            # Reset learning rate for the next iteration
            prev_logli = logli
            learning_rate = initial_learning_rate

    return  true_alpha, pred_alpha, true_theta, pred_theta, iter

# 参数
scale = 1
k = 2               
C = 10000
learning_rate = 0.01
tolerace = 0.00000001
setting = f'k_{k}_scale_{scale}_lrs_{learning_rate}_tor_{tolerace}'

if __name__ == '__main__':
    # env
    number = os.getenv('number')
    np.random.seed(eval(number)) # 设置随机数种子
    iteration = sys.argv[2]
    assert iteration==number

    # parameters
    num_samples = eval(sys.argv[1])
    # k = 2               
    # C = 10000
    # learning_rate = 0.0001
    
    # save npy
    os.makedirs(f'/home/user/CYH/Code_For_MDS/para_result/Normal/{setting}/n_{num_samples}/{number}', exist_ok=True)

    path_ = f'/home/user/CYH/Code_For_MDS/para_result/Normal/{setting}/n_{num_samples}/{number}'
    # 如果文件为空
    if not bool(os.listdir(path_ )):
        result = main(num_samples, k, C, learning_rate/num_samples, tolerace)
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Normal/{setting}/n_{num_samples}/{number}/true_alpha.npy', result[0])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Normal/{setting}/n_{num_samples}/{number}/pred_alpha.npy', result[1])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Normal/{setting}/n_{num_samples}/{number}/true_theta.npy', result[2])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Normal/{setting}/n_{num_samples}/{number}/pred_theta.npy', result[3])

        print(f'n:{num_samples}, number:{number} , iter:{result[-1]}')
    else:
        print(f'n:{num_samples}, number:{number} ---- 已存在')
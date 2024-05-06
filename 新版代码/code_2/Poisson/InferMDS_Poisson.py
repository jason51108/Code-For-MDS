import numpy as np
import matplotlib.pyplot as plt
import random
import time
import statistics
from scipy.special import factorial
import os
import sys

# Poisson生成
def generate_adj(alphas,thetas):
    num_nodes = len(alphas)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            norm_diff = np.linalg.norm(thetas[i] - thetas[j])**2
            m_ij = alphas[i] + alphas[j] - norm_diff
            adjacency_matrix[i][j] = np.random.poisson(lam = np.exp(m_ij)) 
            adjacency_matrix[j][i] = adjacency_matrix[i][j]
    return adjacency_matrix

# fixed
def distance_matrix(point_theta):
    n = point_theta.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(point_theta[i] - point_theta[j])**2
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

# fixed
def generate_M_matrix(alphas, thetas):
    num_nodes = len(alphas)
    M = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            norm_squared = np.linalg.norm(thetas[i] - thetas[j]) ** 2
            M[i][j] = alphas[i] + alphas[j] - norm_squared
            M[j][i] = M[i][j]
    return M

# Poisson生成
def f(A, alphas, thetas):
    M = generate_M_matrix(alphas, thetas)
    num_nodes = A.shape[0]
    result = 0
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                result += -np.exp(M[i][j])-np.log(factorial(A[i][j]))+A[i][j]*M[i][j]
    return result

# fixed
def Projection(X,C):
    return X if np.linalg.norm(X) < C else C*X/np.linalg.norm(X)

# Poisson生成
def gradient_theta(A, alphas, thetas):
    num_nodes = len(alphas)
    gradient = np.zeros_like(thetas)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                norm_squared = np.linalg.norm(thetas[i] - thetas[j]) ** 2
                term = alphas[i] + alphas[j] - norm_squared
                # sigmoid_term = sigmoid(alphas[i] + alphas[j] - norm_squared)

                gradient[i] += -2 * A[i][j] * (thetas[i] - thetas[j]) + \
                                2 * (thetas[i] - thetas[j]) * np.exp(term)*term
    return gradient

# Poisson生成
def gradient_alpha(A, alphas, thetas):
    num_nodes = len(alphas)
    gradient = np.zeros_like(alphas)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                norm_squared = np.linalg.norm(thetas[i] - thetas[j]) ** 2
                term = alphas[i] + alphas[j] - norm_squared
                # sigmoid_term = sigmoid(alphas[i] + alphas[j] - norm_squared)
                gradient[i] += A[i][j] - term*np.exp(term)
    
    return gradient

def main(num_samples, k, C, learning_rate, tolerace, learning_rate_):
    # Generate alpha theta
    true_alpha = np.random.rand(num_samples)
    true_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples,-1)
    # true_theta = np.random.uniform(0, 1, num_samples*k).reshape(num_samples,-1)

    # Generate adjacency matrix
    adjacency_matrix = generate_adj(true_alpha, true_theta)

    # Generate distance matrix
    # distance_matrix_ = distance_matrix(true_theta)
    
    pred_alpha = np.random.rand(num_samples)
    pred_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples,-1)
    # pred_theta = np.random.uniform(0, 1, num_samples*k).reshape(num_samples,-1)

    logli = f(adjacency_matrix, pred_alpha, pred_theta)

    # Learning rate adjustment variables
    decay_rate = 0.9  
    min_learning_rate = learning_rate_

    flag = True
    iter = 0
    while flag:
        iter += 1
        # Update theta
        grad_y = gradient_theta(adjacency_matrix, pred_alpha, pred_theta)
        pred_theta = Projection(pred_theta + learning_rate * grad_y, C)

        # Update alpha
        grad_x = gradient_alpha(adjacency_matrix, pred_alpha, pred_theta)
        pred_alpha = Projection(pred_alpha + learning_rate * grad_x, C)

        # Update and check the log likelihood for convergence
        prev_logli = logli
        logli = f(adjacency_matrix, pred_alpha, pred_theta)

        if (logli - prev_logli)/np.abs(prev_logli) < tolerace:
            flag = False

        # Update the learning rate
        learning_rate = max(min_learning_rate, learning_rate * decay_rate)

    return true_alpha, pred_alpha, true_theta, pred_theta, iter

# 参数
k = 2               
C = 10000
learning_rate = 0.000001
learning_rate_ = 0.000001
tolerace = 0.0001
setting = f'k_{k}_lrs_{learning_rate}_lre_{learning_rate_}_tor_{tolerace}'

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
    # learning_rate = 0.001
    
    # save npy
    os.makedirs(f'/home/user/CYH/Code_For_MDS/para_result/Poisson/{setting}/n_{num_samples}/{number}', exist_ok=True)

    path_ = f'/home/user/CYH/Code_For_MDS/para_result/Poisson/{setting}/n_{num_samples}/{number}'
    # 如果文件为空
    if not bool(os.listdir(path_ )):
        result = main(num_samples, k, C, learning_rate, tolerace, learning_rate_)
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Poisson/{setting}/n_{num_samples}/{number}/true_alpha.npy', result[0])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Poisson/{setting}/n_{num_samples}/{number}/pred_alpha.npy', result[1])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Poisson/{setting}/n_{num_samples}/{number}/true_theta.npy', result[2])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Poisson/{setting}/n_{num_samples}/{number}/pred_theta.npy', result[3])
        print(f'n:{num_samples}, number:{number} , iter:{result[-1]}')
    else:
        print(f'n:{num_samples}, number:{number} ---- 已存在')
    
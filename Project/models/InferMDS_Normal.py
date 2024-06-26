import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit
from utils.tools import *

# Normal分布生成邻接矩阵
def Generate_adj(M, scale):
    np.fill_diagonal(M, 0)
    adjacency_matrix = np.random.normal(loc=M, scale=scale)
    adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T #保证邻接矩阵是对称的
    return adjacency_matrix

# Normal对数似然函数
def Log_likelihood(A, M, scale):
    result = np.sum((A*M - 0.5*M**2)/scale - A**2/(2*scale) - 0.5*np.log(2*np.pi*scale))
    diagonal_correction = np.sum(np.diag((A*M - 0.5*M**2)/scale - A**2/(2*scale) - 0.5*np.log(2*np.pi*scale)))
    
    return result-diagonal_correction

# 对数似然函数对theta的导数
def Gradient_theta(A, M, scale, thetas):
    np.fill_diagonal(M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]  # (num_nodes, num_nodes, num_theta_dims)
    term = (-2 * (diff * A[:, :, np.newaxis]) + 2 * diff * M[:, :, np.newaxis])/scale
    gradient = np.sum(term, axis=1)
    
    return gradient

# 对数似然函数对alpha的导数
def Gradient_alpha(A, M, scale):
    np.fill_diagonal(M, 0)
    gradient = np.sum((A - M)/scale, axis=1)
    
    return gradient

# 主函数
def main(num_samples, k, C, initial_learning_rate, tolerance):
    # 生成真实数据
    true_alpha = np.random.uniform(1, 3, num_samples)
    true_alpha = Centralize_matrix(true_alpha)
    true_theta = truncnorm((-2 - 0) / 1, (2 - 0) / 1, loc=0, scale=1).rvs(size=(num_samples, 2))
    true_theta = Centralize_matrix(true_theta)

    # 生成M矩阵和邻接矩阵
    M = Distance_matrix_M_2(true_alpha, true_theta)
    adjacency_matrix = Generate_adj(M, scale)

    # 生成预测数据
    pred_alpha = np.random.uniform(1, 3, num_samples)
    pred_alpha = Centralize_matrix(pred_alpha)
    pred_theta = truncnorm((-2 - 0) / 1, (2 - 0) / 1, loc=0, scale=1).rvs(size=(num_samples, 2))
    pred_theta = Centralize_matrix(pred_theta)

    # 计算似然函数
    logli = Log_likelihood(adjacency_matrix, M, scale)
    prev_logli = logli

    flag = True
    iter = 0
    learning_rate = initial_learning_rate

    while flag:
        M = Distance_matrix_M_2(pred_alpha, pred_theta)
        grad_y = Gradient_theta(adjacency_matrix, M, scale, pred_theta)
        temp_theta = Projection(pred_theta + learning_rate * grad_y, C)

        M = Distance_matrix_M_2(pred_alpha, temp_theta)
        grad_x = Gradient_alpha(adjacency_matrix, M, scale)
        temp_alpha = Projection(pred_alpha + learning_rate * grad_x, C)

        M = Distance_matrix_M_2(temp_alpha, temp_theta)
        temp_logli = Log_likelihood(adjacency_matrix, M, scale)

        if temp_logli - prev_logli < 0:
            learning_rate /= 5
        else:
            iter += 1
            pred_theta = temp_theta
            pred_alpha = temp_alpha
            logli = temp_logli

            if logli - prev_logli > 0 and (logli - prev_logli) / np.abs(prev_logli) < tolerance:
                flag = False
            prev_logli = logli
            learning_rate = initial_learning_rate

    return true_alpha, pred_alpha, true_theta, pred_theta, adjacency_matrix, iter


k = 2               
C = 10000
scale = 1
learning_rate = 0.1
tolerace = 0.1
setting = f'k_{k}_lr_{learning_rate}_scale_{scale}_tor_{tolerace}'

if __name__ == '__main__':
    # env 
    number = os.getenv('number')
    np.random.seed(eval(number))  # 设置随机数种子
    iteration = sys.argv[2]
    assert iteration==number
    num_samples = eval(sys.argv[1])
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
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Normal/{setting}/n_{num_samples}/{number}/adjacency_matrix.npy', result[4])
        print(f'n:{num_samples}, number:{number} , iter:{result[-1]}')
    else:
        print(f'n:{num_samples}, number:{number} ---- 已存在')
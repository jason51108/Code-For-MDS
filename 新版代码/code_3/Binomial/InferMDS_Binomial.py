import numpy as np
import matplotlib.pyplot as plt
import os
import sys


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成距离矩阵
def distance_matrix(Theta):
    diff = Theta[:, np.newaxis, :] - Theta[np.newaxis, :, :]
    D = np.sum(diff**2, axis=2)
    return D

# 生成矩阵M
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

# 生成邻接矩阵
def generate_adj(alpha, Theta):
    # 计算 m 矩阵
    M = compute_matrix_M(alpha, Theta)

    # # 通过 sigmoid 函数计算连接概率
    p_connect = sigmoid(M)

    # # 生成邻接矩阵
    adjacency_matrix = np.random.binomial(1, p_connect)

    # # 由于邻接矩阵应该是对称的，我们取上三角和下三角的最大值
    adjacency_matrix = np.triu(adjacency_matrix, 1) + np.triu(adjacency_matrix, 1).T
    return adjacency_matrix

# 计算释然函数
def f(A, alpha, Theta):
    M = compute_matrix_M(alpha, Theta)
    sigmoid_M = sigmoid(M)
    
    # 防止对数函数中的参数趋于零
    # epsilon = 1e-10
    # sigmoid_M = np.clip(sigmoid_M, epsilon, 1-epsilon)

    # 计算所有节点对的结果，但需要排除对角线（i == j 的情况）
    result = np.sum(A * M + np.log(1 - sigmoid_M))
    
    # 由于对角线元素被错误地计算了，我们需要从结果中减去这些元素
    diagonal_correction = np.sum(np.diag(A * M + np.log(1 - sigmoid(np.diag(M)))))
    
    return result - diagonal_correction

# 投影函数
def Projection(X,C):
    return X if np.linalg.norm(X) < C else C*X/np.linalg.norm(X)

# 对Thtea求导
def gradient_theta(A, alphas, thetas):
    # 使用已定义的函数计算 M 矩阵
    M = compute_matrix_M(alphas, thetas)
    sigmoid_M = sigmoid(M)
    
    # 计算每个节点对的 theta 差值
    np.fill_diagonal(sigmoid_M, 0)
    diff = thetas[:, np.newaxis, :] - thetas[np.newaxis, :, :]  # (num_nodes, num_nodes, num_theta_dims)
    
    # 计算梯度贡献项
    term = -2 * (diff * A[:, :, np.newaxis]) + 2 * diff * sigmoid_M[:, :, np.newaxis]
    
    # 求和所有贡献以更新梯度
    gradient = np.sum(term, axis=1)
    
    return gradient

# 对Alpha求导
def gradient_alpha(A, alphas, thetas):
    # 使用之前定义的 compute_matrix_M 函数计算 M
    M = compute_matrix_M(alphas, thetas)
    sigmoid_M = sigmoid(M)
    
    # 注意我们需要移除对角线上的元素，因为 i 不等于 j
    np.fill_diagonal(sigmoid_M, 0)
    
    # 每个 alpha[i] 的梯度由所有 j!=i 的 sigmoid_M[i, j] 贡献
    gradient = np.sum(A - sigmoid_M, axis=1)
    
    return gradient


def main(num_samples, k, C, initial_learning_rate, tolerance):
    # Generate true parameters
    true_alpha = np.random.rand(num_samples)
    true_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples, -1)

    adjacency_matrix = generate_adj(true_alpha, true_theta)
    # distance_matrix_true = distance_matrix(true_theta)
    pred_alpha = np.random.rand(num_samples)
    pred_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples, -1)

    # Initial log likelihood
    logli = f(adjacency_matrix, pred_alpha, pred_theta)
    prev_logli = logli  # Initialize previous log likelihood

    # Training loop
    flag = True
    iter = 0
    learning_rate = initial_learning_rate  # Initialize learning rate
    
    # list
    while flag:
        # Update theta
        grad_y = gradient_theta(adjacency_matrix, pred_alpha, pred_theta)
        temp_theta = Projection(pred_theta + learning_rate * grad_y, C)

        # Update alpha
        grad_x = gradient_alpha(adjacency_matrix, pred_alpha, temp_theta)
        temp_alpha = Projection(pred_alpha + learning_rate * grad_x, C)

        # Calculate new log likelihood
        temp_logli = f(adjacency_matrix, temp_alpha, temp_theta)
        
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

k = 2               
C = 10000
learning_rate = 0.1
tolerace = 0.0000001
setting = f'k_{k}_lrs_{learning_rate}_tor_{tolerace}'

if __name__ == '__main__':
    # env 
    number = os.getenv('number')
    # np.random.seed(eval(number)) # 设置随机数种子
    iteration = sys.argv[2]
    assert iteration==number
    num_samples = eval(sys.argv[1])

    # k = 2               
    # C = 10000
    # learning_rate = 0.001
    # tolerace = 0.0001
    # setting = f'k_{k}_lr_{learning_rate}_tor_{tolerace}'

    # save npy
    os.makedirs(f'/home/user/CYH/Code_For_MDS/para_result/Binomial/{setting}/n_{num_samples}/{number}', exist_ok=True)

    path_ = f'/home/user/CYH/Code_For_MDS/para_result/Binomial/{setting}/n_{num_samples}/{number}'
    # 如果文件为空
    if not bool(os.listdir(path_ )):
        result = main(num_samples, k, C, learning_rate/num_samples, tolerace)
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Binomial/{setting}/n_{num_samples}/{number}/true_alpha.npy', result[0])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Binomial/{setting}/n_{num_samples}/{number}/pred_alpha.npy', result[1])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Binomial/{setting}/n_{num_samples}/{number}/true_theta.npy', result[2])
        np.save(f'/home/user/CYH/Code_For_MDS/para_result/Binomial/{setting}/n_{num_samples}/{number}/pred_theta.npy', result[3])
        print(f'n:{num_samples}, number:{number} , iter:{result[-1]}')
    else:
        print(f'n:{num_samples}, number:{number} ---- 已存在')
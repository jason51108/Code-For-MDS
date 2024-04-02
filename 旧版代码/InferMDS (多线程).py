import numpy as np
import matplotlib.pyplot as plt
import random
import time
import statistics
import threading
from concurrent.futures import ThreadPoolExecutor


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def generate_adj(alphas,thetas):
    num_nodes = len(alphas)
    adjacency_matrix = np.zeros((num_nodes, num_nodes))

    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            norm_diff = np.linalg.norm(thetas[i] - thetas[j])**2
            m_ij = alphas[i] + alphas[j] - norm_diff
            adjacency_matrix[i][j] = np.random.binomial(1, sigmoid(m_ij))
            adjacency_matrix[j][i] = adjacency_matrix[i][j]
    return adjacency_matrix

def distance_matrix(point_theta):
    n = point_theta.shape[0]
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(point_theta[i] - point_theta[j])
            distances[i, j] = dist
            distances[j, i] = dist  # 距离矩阵是对称的
    return distances

def generate_M_matrix(alphas, thetas):
    num_nodes = len(alphas)
    M = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            norm_squared = np.linalg.norm(thetas[i] - thetas[j]) ** 2
            M[i][j] = alphas[i] + alphas[j] - norm_squared
            M[j][i] = M[i][j]
    return M

def f(A, alphas, thetas):
    M = generate_M_matrix(alphas, thetas)
    num_nodes = A.shape[0]
    result = 0
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                sigmoid_M = 1 / (1 + np.exp(-M[i][j]))
                result += A[i][j] * M[i][j] + np.log(1 - sigmoid_M)
    return result

def Projection(X,C):
    return X if np.linalg.norm(X) < C else C*X/np.linalg.norm(X)

# 对theta的梯度
def gradient_theta(A, alphas, thetas):
    num_nodes = len(alphas)
    gradient = np.zeros_like(thetas)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                norm_squared = np.linalg.norm(thetas[i] - thetas[j]) ** 2
                exp_term = np.exp(alphas[i] + alphas[j] - norm_squared)
                # sigmoid_term = sigmoid(alphas[i] + alphas[j] - norm_squared)

                gradient[i] += -2 * A[i][j] * (thetas[i] - thetas[j]) + \
                                2 * (thetas[i] - thetas[j]) * exp_term / (1 + exp_term)
    return gradient

# 对alpha的梯度
def gradient_alpha(A, alphas, thetas):
    num_nodes = len(alphas)
    gradient = np.zeros_like(alphas)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                norm_squared = np.linalg.norm(thetas[i] - thetas[j]) ** 2
                exp_term = np.exp(alphas[i] + alphas[j] - norm_squared)
                # sigmoid_term = sigmoid(alphas[i] + alphas[j] - norm_squared)
                gradient[i] += A[i][j] - exp_term / (1 + exp_term)
    
    return gradient

# 单个实验
def main_1(num_samples, k, C, learning_rate):
    # Generate alpha
    random_integers = np.random.rand(num_samples)

    # # Generate theta method 1
    # random_vectors = np.random.rand(num_samples, k)

    # Generate theta method 2
    # random_vectors = np.random.uniform(0, 45*np.pi/96, num_samples*k).reshape(num_samples,-1)

    # Generate theta method 3
    random_vectors = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples,-1)

    # Generate adjacency matrix
    adjacency_matrix = generate_adj(random_integers, random_vectors)

    # Generate distance matrix
    distance_matrix_ = distance_matrix(random_vectors)
    

    point_alpha = np.random.rand(num_samples)
    point_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples,-1)

    logli = f(adjacency_matrix, point_alpha, point_theta)

    hahaha = []
    loss_alpha = []
    loss_theta = []

    flag = True

    # for _ in range(num_iterations):
    while flag:
        # 更新theta
        grad_y = gradient_theta(adjacency_matrix, point_alpha, point_theta)
        prev_theta = point_theta
        point_theta = Projection(point_theta + learning_rate*grad_y, C)
        # loss_theta.append(np.linalg.norm(distance_matrix_-distance_matrix(point_theta),ord='fro'))
        loss_theta.append(np.linalg.norm(distance_matrix_-distance_matrix(point_theta), ord='fro')**2/len(point_theta)**2)

        # 更新alpha
        grad_x = gradient_alpha(adjacency_matrix, point_alpha, point_theta)
        prev_alpha = point_alpha
        point_alpha = Projection(point_alpha + learning_rate*grad_x, C)
        # loss_alpha.append(np.linalg.norm(random_integers-point_alpha))
        loss_alpha.append(np.linalg.norm(random_integers-point_alpha)**2/len(point_alpha))
        
        # 更新负对数似然函数
        hahaha.append(logli)
        prev_logli = logli
        logli = f(adjacency_matrix, point_alpha, point_theta)

        # 相对误差
        # if (logli-prev_logli)/np.abs(prev_logli) < np.exp(-6):
        #     flag = False

        # 绝对误差
        if (logli-prev_logli) < 1/100:
            flag = False

    return loss_alpha[-1], loss_theta[-1]

# repeat实验
def main_2(num, repeat, num_samples, k, C, learning_rate):
    loss_ = []
    for _ in range(repeat):
        loss_.append(main_1(num_samples, k, C, learning_rate))
    # output
    alpha_loss_temp = [_[0] for _ in loss_]
    theta_loss_temp = [_[-1] for _ in loss_]
    with open(fr'C:\Users\陈颖航\Desktop\overleaf论文\txt\output_{num}.txt', 'a+') as ff:
        ff.write(f'''When repeat={repeat},
                for alpha_loss: \t25%:{round(statistics.quantiles(alpha_loss_temp, n=4)[0],4)}, \t75%:{round(statistics.quantiles(alpha_loss_temp, n=4)[-1],4)}, \tmean:{round(statistics.mean(alpha_loss_temp),4)}
                for theta_loss: \t25%:{round(statistics.quantiles(theta_loss_temp, n=4)[0],4)}, \t75%:{round(statistics.quantiles(theta_loss_temp, n=4)[-1],4)}, \tmean:{round(statistics.mean(theta_loss_temp),4)}\n''')
        
        ff.write('\talpha_loss:[')
        for item in alpha_loss_temp:
            ff.write(str(round(item,4))+',\t')
        ff.write(']\n')

        ff.write('\ttheta_loss:[')
        for item in theta_loss_temp:
            ff.write(str(round(item,4))+',\t')
        ff.write(']\n')
    return loss_

# 线程函数
def thread_function(num, repeat, k, C, learning_rate, loss_total):
    # 假设 main_2 是你的主要计算函数
    loss = main_2(num, repeat, num, k, C, learning_rate)
    # 在这里，我们需要同步访问 loss_total
    with threading.Lock():
        loss_total.append(loss)

if __name__ == '__main__':
    start_time = time.time()

    # 使用线程安全的列表
    loss_total = []

    # 参数设定
    repeat = 10
    k = 2
    C = 10000
    learning_rate = 0.01

    train_ = [10, 30, 50, 80, 100, 150, 200]  # 可以添加更多的训练参数

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(thread_function, num, repeat, k, C, learning_rate, loss_total) for num in train_]
        for future in futures:
            future.result()  # 等待所有线程完成

    # total time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time//60} mins {total_time%60} seconds")

    # 画图
    alpha_loss = [_[0] for _ in loss_total]
    theta_loss = [_[-1] for _ in loss_total]

    # plt.boxplot(alpha_loss, positions=np.arange(len(train_))*2-0.4, widths=0.4, labels=train_, patch_artist=True)
    # plt.boxplot(theta_loss, positions=np.arange(len(train_))*2+0.4, widths=0.4, labels=train_, patch_artist=True)

    # 画 alpha_loss 的箱线图
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.boxplot(alpha_loss, labels=train_)
    plt.title(fr'Boxplot Comparison for the loss of $\alpha$')
    plt.xlabel('N')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig(r'C:\Users\陈颖航\Desktop\overleaf论文\image\alpha_loss.png')
    plt.close('all') #清空缓存

    # 画 theta_loss 的箱线图
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.boxplot(theta_loss, labels=train_)
    plt.title(fr'Boxplot Comparison for the loss of $\theta$')
    plt.xlabel('N')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig(r'C:\Users\陈颖航\Desktop\overleaf论文\image\theta_loss.png')
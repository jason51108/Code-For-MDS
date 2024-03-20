import numpy as np
import matplotlib.pyplot as plt
import random
import threading
import time

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


def worker(num_samples, k, C, learning_rate, initial_point_1, initial_point_2, results):
    loss_alpha, loss_theta = main(num_samples, k, C, initial_point_1, initial_point_2, learning_rate)
    results[num_samples].append((loss_alpha, loss_theta))


def main(num_samples, k, C, initial_point_1, initial_point_2, learning_rate):
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
    
    point_alpha= initial_point_1
    point_theta= initial_point_2

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
        # loss_theta.append(np.linalg.norm(distance_matrix_-distance_matrix(point_theta),ord='nuc'))
        loss_theta.append(np.linalg.norm(distance_matrix_-distance_matrix(point_theta),ord='nuc')**2/len(initial_point_2)**2)

        # 更新alpha
        grad_x = gradient_alpha(adjacency_matrix, point_alpha, point_theta)
        prev_alpha = point_alpha
        point_alpha = Projection(point_alpha + learning_rate*grad_x, C)
        # loss_alpha.append(np.linalg.norm(random_integers-point_alpha))
        loss_alpha.append(np.linalg.norm(random_integers-point_alpha)**2/len(initial_point_1))
        
        # 更新负对数似然函数
        hahaha.append(logli)
        prev_logli = logli
        logli = f(adjacency_matrix, point_alpha, point_theta)

        # 相对误差
        # if (logli-prev_logli)/np.abs(prev_logli) < np.exp(-6):
        #     flag = False

        # 绝对误差
        if (logli-prev_logli) < np.exp(-1):
            flag = False

    return loss_alpha[-1], loss_theta[-1]




if __name__ == '__main__':
    start_time = time.time()
    train_ = [10, 20, 50, 80, 100, 200, 500]

    # 创建线程池
    threads = []

    # 创建一个字典用于存储每个训练数量对应的结果
    results = {num: [] for num in train_}


    for num_samples in train_:
        print(f"Running with {num_samples} samples...")

        for _ in range(10):
            k = 2
            C = 1.5
            learning_rate = 0.001

            initial_point_1 = np.random.rand(num_samples)
            initial_point_2 = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples,-1)

            # 创建线程并将其加入线程池
            thread = threading.Thread(target=worker, args=(num_samples, k, C, learning_rate, initial_point_1, initial_point_2, results))
            threads.append(thread)
            thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 计算总的运行时间
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
    plt.savefig('./alpha_loss.png')
    plt.close('all') #清空缓存

    # 画 theta_loss 的箱线图
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.boxplot(theta_loss, labels=train_)
    plt.title(fr'Boxplot Comparison for the loss of $\theta$')
    plt.xlabel('N')
    plt.ylabel('Loss')
    # plt.show()
    plt.savefig('./theta_loss.png')

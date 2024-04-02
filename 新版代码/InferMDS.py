import numpy as np
import time
import statistics


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


def find_optimal_Q(A, B):
    # Compute the matrix C as the product of A and the transpose of B
    C = A @ B.T
    
    # Perform Singular Value Decomposition (SVD) on C
    U, _, Vt = np.linalg.svd(C)
    
    # Compute the optimal Q as the product of U and V^T
    Q = U @ Vt
    
    return Q


def centralize_matrix(X):
    # Calculate the mean of each column
    column_means = X.mean(axis=0)
    
    # Subtract the column means from the respective columns to centralize the matrix
    X_centered = X - column_means
    
    return X_centered


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

    # Generate theta
    random_vectors = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples,-1)

    # Generate adjacency matrix
    adjacency_matrix = generate_adj(random_integers, random_vectors)

    # Generate distance matrix
    distance_matrix_ = distance_matrix(random_vectors)
    
    # Generate init points
    point_alpha = np.random.rand(num_samples)
    point_theta = np.random.uniform(-45*np.pi/161, 45*np.pi/161, num_samples*k).reshape(num_samples,-1)

    # cal logli
    logli = f(adjacency_matrix, point_alpha, point_theta)


    # find the opt-para
    flag = True
    while flag:
        # 更新theta
        grad_y = gradient_theta(adjacency_matrix, point_alpha, point_theta)
        # prev_theta = point_theta
        point_theta = Projection(point_theta + learning_rate*grad_y, C)
        # loss_theta.append(np.linalg.norm(distance_matrix_-distance_matrix(point_theta), ord='fro')**2/len(point_theta)**2)

        # 更新alpha
        grad_x = gradient_alpha(adjacency_matrix, point_alpha, point_theta)
        # prev_alpha = point_alpha
        point_alpha = Projection(point_alpha + learning_rate*grad_x, C)
        # loss_alpha.append(np.linalg.norm(random_integers-point_alpha)**2/len(point_alpha))
        
        # 更新负对数似然函数
        prev_logli = logli
        logli = f(adjacency_matrix, point_alpha, point_theta)

        # 绝对误差
        if (logli-prev_logli) < 0.01:
            flag = False

    return random_integers, random_vectors.reshape(-1), point_alpha, point_theta.reshape(-1)


# repeat实验
# def main_2(num, repeat, num_samples, k, C, learning_rate):
#     print(f'num for node ===== {num}')
#     result_ = []
#     for _ in range(repeat):
#         result_.append(main_1(num_samples, k, C, learning_rate))

#     # output
#     origin_alpha = np.array([_[0] for _ in result_])
#     origin_theta = np.array([_[1] for _ in result_])
#     result_alpha = np.array([_[2] for _ in result_])
#     result_theta = np.array([_[3] for _ in result_])
#     origin_theta_centered = centralize_matrix(origin_theta)
#     result_theta_centered = centralize_matrix(result_theta)

#     # alpha-boxplot
#     box_alpha = (origin_alpha-origin_theta)**2/len(result_alpha)

#     # distance

#     # 任务一
#     alpha_loss_mean = np.linalg.norm(result_alpha-origin_alpha)**2/len(result_alpha)

#     # 任务二
#     alpha_loss_max = np.abs((result_alpha-origin_alpha)).max()

#     # 任务三
#     distance_loss = (np.linalg.norm((distance_matrix(origin_theta) - distance_matrix(result_theta)), ord='fro')**2)/(len(result_theta)**2)
    
#     # 任务四
#     pass

#     # 任务五
#     Q_matrix = find_optimal_Q(origin_theta, result_theta)
#     theta_loss = (np.linalg.norm((Q_matrix@result_theta_centered-origin_theta_centered), ord='fro')**2)

    

    # 写入文件
    # with open(fr'C:\Users\陈颖航\Desktop\overleaf论文\txt\{setting}_output_{num}.txt', 'a+') as ff:
    #     ff.write(f'''When repeat={repeat},
    #             for alpha_loss: \t25%:{round(statistics.quantiles(alpha_loss_temp, n=4)[0],4)}, \t75%:{round(statistics.quantiles(alpha_loss_temp, n=4)[-1],4)}, \tmean:{round(statistics.mean(alpha_loss_temp),4)}
    #             for theta_loss: \t25%:{round(statistics.quantiles(theta_loss_temp, n=4)[0],4)}, \t75%:{round(statistics.quantiles(theta_loss_temp, n=4)[-1],4)}, \tmean:{round(statistics.mean(theta_loss_temp),4)}\n''')
        
    #     ff.write('\talpha_loss:[')
    #     for item in alpha_loss_temp:
    #         ff.write(str(round(item,4))+',\t')
    #     ff.write(']\n')

    #     ff.write('\ttheta_loss:[')
    #     for item in theta_loss_temp:
    #         ff.write(str(round(item,4))+',\t')
    #     ff.write(']\n')

    return alpha_loss_mean, alpha_loss_max, distance_loss, theta_loss



if __name__ == '__main__':
    # start_time = time.time()

    # # create a list to memory loss
    # loss_total = []

    # 参数设定
    repeat = 20
    num_samples = 10
    k = 2               
    C = 10000
    learning_rate = 0.01
    result = main_1(num_samples, k, C, learning_rate)[1]
    np.save('result.npy', result)
    # setting = 'k_{}_C_{}_lr{}_repeat_{}'.format(k,C,learning_rate,repeat)

    # train_ = [10, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1000] 
    # for idx, num in enumerate(train_):
    #     print(f"Running iteration {idx+1}/{len(train_)}...")
    #     loss_total.append(main_2(num, repeat, num, k, C, learning_rate))
    
    # # total time
    # end_time = time.time()
    # total_time = end_time - start_time
    # print(f"Total running time: {total_time//60} mins {total_time%60} seconds")

    # # 画图
    # alpha_loss_mean_ = [_[0] for _ in loss_total]
    # alpha_loss_max_ = [_[1] for _ in loss_total]
    # distance_loss_ = [_[2] for _ in loss_total]
    # theta_loss_ = [_[-1] for _ in loss_total]



    # # 画 alpha_loss_mean_ 的折线图
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.plot(alpha_loss_mean_)
    # plt.title(fr'plot for the mean_loss of $\alpha$')
    # plt.xlabel('N')
    # plt.ylabel('Loss')
    # # plt.show()
    # plt.savefig(fr'C:\Users\陈颖航\Desktop\overleaf论文\image\{setting}_alpha_loss_mean.png')
    # plt.close('all') #清空缓存

    # # 画出alpha_loss_max_的折线图
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.plot(alpha_loss_max_)
    # plt.title(fr'plot for the max_loss of $\alpha$')
    # plt.xlabel('N')
    # plt.ylabel('Loss')
    # # plt.show()
    # plt.savefig(fr'C:\Users\陈颖航\Desktop\overleaf论文\image\{setting}_alpha_loss_max.png')
    # plt.close('all') #清空缓存

    # # 画 distance_loss_ 的折现图
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.plot(distance_loss_)
    # plt.title(fr'plot for the distance_loss of distacne')
    # plt.xlabel('N')
    # plt.ylabel('Loss')
    # # plt.show()
    # plt.savefig(fr'C:\Users\陈颖航\Desktop\overleaf论文\image\{setting}_distance_loss.png')
    # plt.close('all') #清空缓存

    # # 画 theta_loss_ 的折现图
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['top'].set_visible(False)
    # plt.plot(theta_loss_)
    # plt.title(fr'plot for the theta_loss of $\theta$')
    # plt.xlabel('N')
    # plt.ylabel('Loss')
    # # plt.show()
    # plt.savefig(fr'C:\Users\陈颖航\Desktop\overleaf论文\image\{setting}_theta_loss.png')
    # plt.close('all') #清空缓存
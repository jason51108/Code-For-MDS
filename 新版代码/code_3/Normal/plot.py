import numpy as np
import os
import matplotlib.pyplot as plt
import InferMDS_Normal as Ib


# 生成距离矩阵
def distance_matrix_3d(point_theta):
    k, b, _ = point_theta.shape
    distances = np.zeros((k, b, b))

    for idx in range(k):
        diff = point_theta[idx, :, np.newaxis, :] - point_theta[idx, np.newaxis, :, :]
        distances[idx] = np.sum(diff**2, axis=-1)

    return distances


# 计算平均frobenius范数
def average_frobenius_norm_squared(dist_matrix1, dist_matrix2):
    # 计算两个矩阵的差
    diff_matrix = dist_matrix1 - dist_matrix2
    # 计算 F-范数的平方并进行标准化
    norm_squared = np.sum(np.square(diff_matrix), axis=(1, 2)) / (dist_matrix1.shape[1] ** 2)
    
    return norm_squared


def frobenius_norm_squared(dist_matrix1, dist_matrix2):
    # 计算两个矩阵的差
    diff_matrix = dist_matrix1 - dist_matrix2
    # 计算 F-范数的平方
    norm_squared = np.sum(np.square(diff_matrix), axis=(1, 2))
    
    return norm_squared


# 矩阵中心化
def maximize_centered(Theta, Theta_tilde):
    # 获取输入的维度
    a, b, c = Theta.shape
    
    # 中心化 Theta 和 Theta_tilde
    Theta_mean = Theta.mean(axis=1, keepdims=True)
    Theta_tilde_mean = Theta_tilde.mean(axis=1, keepdims=True)
    
    Theta_ = Theta - Theta_mean
    Theta_tilde_ = Theta_tilde - Theta_tilde_mean

    return Theta_, Theta_tilde_




# 计算最优矩阵Q
def maximize_trace_3d(Theta_star, Theta_tilde_star):
    a, b, c = Theta_star.shape  # 获取输入的维度
    Q_matrices = np.zeros((a, c, c))  # 初始化存储 Q 矩阵的数组

    for i in range(a):
        # 计算中心化矩阵的乘积
        Theta_product = Theta_tilde_star[i].T@ Theta_star[i]

        # 奇异值分解
        U, _, VT = np.linalg.svd(Theta_product, full_matrices=True)

        # 构造正交矩阵 Q
        Q = VT.T @ U.T

        # 存储每次实验的 Q 矩阵
        Q_matrices[i] = Q

    return Q_matrices


# 计算theta与Q的乘积
def transform_space(data, Q_matrices):
    # 使用 einsum 进行批量矩阵乘法
    transformed_data = np.einsum('ijk,ikl->ijl', data, Q_matrices.transpose(0, 2, 1))
    return transformed_data

# 生成单个n的集合
def stack_arrays(base_path):
    filenames = ["pred_alpha", "true_alpha", "pred_theta", "true_theta"]
    results = {filename: [] for filename in filenames}

    subfolders = sorted([os.path.join(base_path, d) for d in os.listdir(base_path)], key=lambda x: int(os.path.basename(x)))
    
    for subfolder in subfolders:
        for filename in filenames:
            file_path = os.path.join(subfolder, filename + ".npy")
            if os.path.exists(file_path):
                data = np.load(file_path)
                if data.ndim == 1:
                    data = data[np.newaxis, :]
                elif data.ndim == 2:
                    data = data[np.newaxis, :, :]
                results[filename].append(data)

    for filename in filenames:
        if results[filename]:  # 确保列表不为空
            results[filename] = np.concatenate(results[filename], axis=0)
    
    return results


# 计算各损失
def calculate_losses(base_folder, n_values):
    losses_dict = {
        'Avg loss of alpha': {},
        'Max loss of alpha': {},
        'Avg F_norm of distance': {},
        'Max distance':{},
        'Avg F_norm of theta': {},
        'two_to_infty of theta':{}
    }

    for n in n_values:
        folder_path = os.path.join(base_folder, "n_" + str(n))
        result = stack_arrays(folder_path)

        # Task 1: Alpha losses
        alpha_losses = np.linalg.norm(result['pred_alpha'] - result['true_alpha'], axis=1) ** 2 / result['pred_alpha'].shape[-1]
        losses_dict['Avg loss of alpha'][str(n)] = alpha_losses

        # Task 2: Max alpha difference
        max_alpha_diff = np.abs(result['pred_alpha'] - result['true_alpha']).max(axis=1)
        losses_dict['Max loss of alpha'][str(n)] = max_alpha_diff

        # Task 3: Average Frobenius norm squared for true and pred theta
        true_matrix = distance_matrix_3d(result['true_theta'])
        pred_matrix = distance_matrix_3d(result['pred_theta'])
        avg_frobenius = average_frobenius_norm_squared(true_matrix, pred_matrix)
        losses_dict['Avg F_norm of distance'][str(n)] = avg_frobenius

        # Task 4: Max Frobenius norm squared for true and pred theta
        max_distance = np.array([_.max() for _ in np.abs(true_matrix-pred_matrix)])
        losses_dict['Max distance'][str(n)] = max_distance


        # Task 5: Frobenius norm squared after transformation
        Theta_star, Theta_tilde_star = maximize_centered(result['true_theta'], result['pred_theta'])
        Q = maximize_trace_3d(Theta_star, Theta_tilde_star)
        avg_frobenius_transformed = average_frobenius_norm_squared(transform_space(Theta_tilde_star, Q), Theta_star)
        losses_dict['Avg F_norm of theta'][str(n)] = avg_frobenius_transformed

        # Task6: two to infty
        two_to_infty = [np.linalg.norm(_,axis=1).max() for _ in np.abs(transform_space(Theta_tilde_star, Q) - Theta_star)]
        losses_dict['two_to_infty of theta'][str(n)] = two_to_infty

    return losses_dict

# 画图
def plot_boxplot(data_dict, setting, title='Losses Boxplot'):
    plt.figure(figsize=(15, 9))
    for idx, (key, value) in enumerate(data_dict.items(), start=1):
        plt.subplot(2, 3, idx)
        plt.boxplot(value.values())
        plt.xticks(range(1, len(value) + 1), value.keys())
        plt.xlabel('N')
        plt.ylabel('Loss')
        plt.title(f'{key} Boxplot')
        plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(rf'/home/user/CYH/Code_For_MDS/image/Normal/{setting}.png')

if __name__ == '__main__':
    # Calculate losses and plot boxplots
    # os.chdir(r'/data/Test_CYH/code')
    os.makedirs(f'/home/user/CYH/Code_For_MDS/image/Normal/', exist_ok=True)
    base_folder = rf'/home/user/CYH/Code_For_MDS/para_result/Normal/{Ib.setting}'
    # n_values = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    n_values = [200, 400, 600, 800, 1000]
    # n_values = [50, 80, 100, 150, 200, 300]
    # n_values = [10, 30, 50, 80, 100, 150, 200]
    losses_dict = calculate_losses(base_folder, n_values)
    plot_boxplot(losses_dict, Ib.setting)
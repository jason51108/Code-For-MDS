import numpy as np
import time
import os
import matplotlib.pyplot as plt


# 进行批量矩阵距离计算
def distance_matrix_3d(point_theta):
    k, b, _ = point_theta.shape
    distances = np.zeros((k, b, b))

    for idx in range(k):
        diff = point_theta[idx, :, np.newaxis, :] - point_theta[idx, np.newaxis, :, :]
        distances[idx] = np.sum(diff**2, axis=-1)

    return distances

# 进行批量矩阵乘法
def transform_array(arr):
    result = np.matmul(arr, arr.transpose(0, 2, 1))

    return result

# 计算平均frobenius范数
def average_frobenius_norm_squared(dist_matrix1, dist_matrix2):
    diff_matrix = dist_matrix1 - dist_matrix2
    norm_squared = np.sum(np.square(diff_matrix), axis=(1, 2)) / (dist_matrix1.shape[1] ** 2)

    return norm_squared

# 计算frobenius范数
def frobenius_norm_squared(dist_matrix1, dist_matrix2):
    diff_matrix = dist_matrix1 - dist_matrix2
    norm_squared = np.sum(np.square(diff_matrix), axis=(1, 2))
    
    return norm_squared

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

def plot_boxplot(data_dict, setting, title='Boxplot of losses'):
    plt.figure(figsize=(16, 10))
    num_plots = len(data_dict)
    
    for idx, (key, value) in enumerate(data_dict.items(), start=1):
        plt.subplot(2, 2, idx)
        # plt.subplot(1, num_plots, idx)
        plt.boxplot(value.values())
        plt.xticks(range(1, len(value) + 1), value.keys())
        plt.xlabel('N')
        plt.ylabel('Loss')
        plt.title(f'{key} Boxplot')
        # plt.grid(True)
        
    plt.tight_layout()
    # plt.show()
    folder_path = '/home/user/CYH/Code_For_MDS/Project/figure'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(folder_path + f'/{setting}.png')


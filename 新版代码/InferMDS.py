import numpy as np
import matplotlib.pyplot as plt
import random
import time
import statistics
import os
import sys


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
            dist = np.linalg.norm(point_theta[i] - point_theta[j])**2
            distances[i, j] = dist
            distances[j, i] = dist
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
                sigmoid_M = sigmoid(M[i][j])
                result += A[i][j] * M[i][j] + np.log(1 - sigmoid_M)
    return result

def Projection(X,C):
    return X if np.linalg.norm(X) < C else C*X/np.linalg.norm(X)

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

def main(num_samples, k, C, learning_rate):
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
    min_learning_rate = 0.001

    flag = True
    while flag:
        # Update theta
        grad_y = gradient_theta(adjacency_matrix, pred_alpha, pred_theta)
        pred_theta = Projection(pred_theta + learning_rate * grad_y, C)

        # Update alpha
        grad_x = gradient_alpha(adjacency_matrix, pred_alpha, pred_theta)
        pred_alpha = Projection(pred_alpha + learning_rate * grad_x, C)

        # Update and check the log likelihood for convergence
        prev_logli = logli
        logli = f(adjacency_matrix, pred_alpha, pred_theta)

        if (logli - prev_logli)/np.abs(prev_logli) < 0.0001:
            flag = False

        # Update the learning rate
        learning_rate = max(min_learning_rate, learning_rate * decay_rate)

    return true_alpha, pred_alpha, true_theta, pred_theta


if __name__ == '__main__':
    # env
    number = os.getenv('number')
    iteration = sys.argv[2]
    assert iteration==number

    # parameters
    num_samples = eval(sys.argv[1])
    k = 2               
    C = 10000
    learning_rate = 0.001
    
    # save npy
    os.makedirs(f'../para_result/n_{num_samples}/{number}', exist_ok=True)

    result = main(num_samples, k, C, learning_rate)
    np.save(f'../para_result/n_{num_samples}/{number}/true_alpha.npy', result[0])
    np.save(f'../para_result/n_{num_samples}/{number}/pred_alpha.npy', result[1])
    np.save(f'../para_result/n_{num_samples}/{number}/true_theta.npy', result[2])
    np.save(f'../para_result/n_{num_samples}/{number}/pred_theta.npy', result[3])
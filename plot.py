import numpy as np
import time
import os
import matplotlib.pyplot as plt
import argparse
from utils.loss import stack_arrays, transform_array, distance_matrix_3d, average_frobenius_norm_squared, plot_boxplot


# 计算各损失
def plot(args, setting):
        base_folder = (f'/home/user/CYH/Code_For_MDS/Project/para_result/{args.task_name}/'
                        f'{args.type}/{args.model}/'
                        f'{args.constrain}_{args.dimension}_{args.learning_rate}'
                        f'{args.patience}_{args.tolerace}/')
        n_values = [eval(i.split('_')[-1]) for i in os.listdir(base_folder)]
        n_values.sort()
        
        # losses
        losses_dict = {
            'Avg loss of alpha': {},
            'Max loss of alpha': {},
            'Avg F_norm loss of matrix': {},
            'Max loss of matrix':{}
        }

        for n in n_values:
            folder_path = os.path.join(base_folder, "n_" + str(n))
            result = stack_arrays(folder_path)

            # Task 1: Alpha losses
            alpha_losses = np.linalg.norm(result['pred_alpha']-result['true_alpha'], axis=1) ** 2 / (result['pred_alpha'].shape[-1])
            losses_dict['Avg loss of alpha'][str(n)] = alpha_losses

            # Task 2: Max alpha difference
            max_alpha_diff = np.abs(result['pred_alpha'] - result['true_alpha']).max(axis=1)
            losses_dict['Max loss of alpha'][str(n)] = max_alpha_diff

            # Task 3: Avg F_norm loss of matrix
            if args.type == 'inner-product':
                pred_matrix = transform_array(result['pred_theta'])
                true_matrix = transform_array(result['true_theta'])
                avg_frobenius_transformed = average_frobenius_norm_squared(true_matrix, pred_matrix)
                losses_dict['Avg F_norm loss of matrix'][str(n)] = avg_frobenius_transformed
                # Task 4: Avg F_norm loss of matrix
                max_distance = np.array([_.max() for _ in np.abs(pred_matrix-true_matrix)])
                losses_dict['Max loss of matrix'][str(n)] = max_distance

            elif args.type in ['distance1', 'distance2']:
                pred_matrix = distance_matrix_3d(result['pred_theta'])
                true_matrix = distance_matrix_3d(result['true_theta'])
                avg_frobenius_transformed = average_frobenius_norm_squared(true_matrix, pred_matrix)
                losses_dict['Avg F_norm loss of matrix'][str(n)] = avg_frobenius_transformed
                # Task 4: Avg F_norm loss of matrix
                max_distance = np.array([_.max() for _ in np.abs(pred_matrix-true_matrix)])
                losses_dict['Max loss of matrix'][str(n)] = max_distance

            # # Task 5: Frobenius norm squared after transformation
            # avg_frobenius_transformed = average_frobenius_norm_squared(result['true_theta'], result['pred_theta'])
            # losses_dict['Avg F_norm of theta'][str(n)] = avg_frobenius_transformed

            # # Task6: two to infty
            # two_to_infty = [np.linalg.norm(_,axis=1).max() for _ in np.abs(result['true_theta'] - result['pred_theta'])]
            # losses_dict['two_to_infty of theta'][str(n)] = two_to_infty
            
        plot_boxplot(losses_dict, setting)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Statistical Inference on the Euclidean Embedding for Network Data')

    # Basic config
    parser.add_argument('--task_name', type=str, default='parameter estimation',
                    help='task name, options:[parameter estimation, classification, matrix completion, ......]')
    parser.add_argument('--type', type=str, default='inner-product', help='model type, options: [distance1, distance2, inner-product]')
    parser.add_argument('--model', type=str, default='Binomial', help='Model name, options: [Binomial, Poisson, Normal]')
    
    # parameter
    parser.add_argument('--constrain', type=int, default=10000, help='projection limit value')
    parser.add_argument('--dimension', type=int, default=2, help='model dimension')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1.0, help='optimizer learning rate')
    parser.add_argument('--patience', type=str, default='relative', help='likelihood function increases proportion, options: [relative, absolute]')
    parser.add_argument('--tolerace', type=float, default=0.0000000001, help='tolerace of patience')

    args = parser.parse_args()
    setting = '{}_{}_{}_c{}_k{}_lr{}_pt{}_tl{}'.format(
        args.task_name,
        args.type,
        args.model,
        args.constrain,
        args.dimension,
        args.learning_rate,
        args.patience,
        args.tolerace)

    start_time = time.time()
    plot(args, setting)
    end_time = time.time()
    execution_time = end_time - start_time
    print("time：", execution_time, "s")
import os
import random
import argparse
import numpy as np

from exp.exp_parameter_estimation import Exp_Parameter_Estimation
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if __name__ == '__main__':
    # random number
    fix_seed = 2024
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    
    # Task name
    parser = argparse.ArgumentParser(description='Statistical Inference on the Euclidean Embedding for Network Data')

    # Basic config
    parser.add_argument('--task_name', type=str, default='parameter estimation',
                    help='task name, options:[parameter estimation, classification, matrix completion, ......]') # , required=True
    parser.add_argument('--type', type=str, default='distance1', help='model type, options: [distance1, distance2, inner-product]') # , required=True
    parser.add_argument('--model', type=str, default='Poisson', help='Model name, options: [Binomial, Poisson, Normal]') # , required=True
    
    # parameter
    parser.add_argument('--seed_number', type=int, default=1, help='Random number seed') #, required=True
    parser.add_argument('--num_samples', type=int, default=236, help='the number of repetitions of the experiment') # , required=True
    parser.add_argument('--constrain', type=int, default=10000, help='projection limit value')
    parser.add_argument('--dimension', type=int, default=2, help='model dimension')

    # data loader
    parser.add_argument('--data', type=str, default='Custom', help='options:[Simulation, Custom]') # , required=True
    parser.add_argument('--root_path', type=str, default='./data_provider/dataset/') #/home/user/CYH/Code_For_MDS/Project
    parser.add_argument('--data_path', type=str, default='adjacency_matrix.npy')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='optimizer learning rate')
    parser.add_argument('--patience', type=str, default='relative', help='likelihood function increases proportion, options: [relative, absolute]')
    parser.add_argument('--tolerace', type=float, default=0.01, help='tolerace of patience')

    # especially for normal distributions
    parser.add_argument('--scale', type=float, default=1.0, help='variance of normal distribution')
    
    # especially for distance1
    parser.add_argument('--rho', type=float, default=0, help='rho of distance1')
    
    args = parser.parse_args()

    # print
    # print('Args in experiment:')
    # print(args)

    # choose task
    if args.task_name == 'parameter estimation':
        Exp = Exp_Parameter_Estimation
    elif args.task_name == 'classification':
        pass
    elif args.task_name == 'matrix completion':
        pass
    else:
        raise ValueError("task_name must be in [parameter estimation, classification, matrix completion]")

    exp = Exp(args)  # set experiments
    # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train()
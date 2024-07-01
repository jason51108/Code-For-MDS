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
    parser.add_argument('--task_name', type=str, required=True, default='parameter estimation',
                    help='task name, options:[parameter estimation, classification, matrix completion, ......]')
    parser.add_argument('--type', type=str, required=True, default='inner-product', help='model type, options: [distance1, distance2, inner-product]')
    parser.add_argument('--model', type=str, required=True, default='Binomial', help='Model name, options: [Binomial, Poisson, Normal]')
    
    # parameter
    parser.add_argument('--seed_number', type=int, required=True, default=1, help='Random number seed') # ssually set in script
    parser.add_argument('--num_samples', type=int, required=True, default=500, help='the number of repetitions of the experiment')
    parser.add_argument('--constrain', type=int, default=10000, help='projection limit value')
    parser.add_argument('--dimension', type=int, default=2, help='model dimension')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1, help='optimizer learning rate')
    parser.add_argument('--patience', type=str, default='relative', help='likelihood function increases proportion, options: [relative, absolute]')
    parser.add_argument('--tolerace', type=float, default=0.01, help='tolerace of patience')

    args = parser.parse_args()

    # print
    # print('Args in experiment:')
    # print(args)

    # choose task
    if args.task_name == 'parameter estimation':
        Exp = Exp_Parameter_Estimation
    elif args.task_name == 'estimation':
        pass
    elif args.task_name == 'matrix completion':
        pass
    else:
        pass

    exp = Exp(args)  # set experiments
    # print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train()
    

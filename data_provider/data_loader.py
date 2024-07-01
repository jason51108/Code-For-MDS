import numpy as np
import warnings
from scipy.stats import truncnorm
from scipy.special import expit
from utils.tools import Centralize_matrix, Generate_adj_binomial, Distance_matrix_M_1, Distance_matrix_M_2, Inner_matrix_M
warnings.filterwarnings('ignore')


def Binomial_data(args):
    true_alpha = np.random.uniform(1, 3, args.num_samples)
    true_alpha = Centralize_matrix(true_alpha)
    true_theta = truncnorm((-2 - 0) / 1, (2 - 0) / 1, loc=0, scale=1).rvs(size=(args.num_samples, args.dimension))
    true_theta = Centralize_matrix(true_theta)
    
    np.random.seed(args.seed_number) # Ensure that the true parameters are the same but the adjacency matrix is different.
    # Choose M
    if args.type == 'distance1':
        Generate_M = Distance_matrix_M_1
    elif args.type == 'distance2':
        Generate_M = Distance_matrix_M_2
    elif args.type == 'inner-product':
        Generate_M = Inner_matrix_M
    else:
        raise ValueError("type must be distance1, distance2 or inner-product")
    
    M = Generate_M(true_alpha, true_theta)
    adjacency_matrix = Generate_adj_binomial(M)
    
    return true_alpha, true_theta, adjacency_matrix
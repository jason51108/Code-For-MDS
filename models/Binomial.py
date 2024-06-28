import os
import argparse
import numpy as np
from scipy.stats import truncnorm
from scipy.special import expit
from utils.tools import *


class Model():
    def __init__(self, args):
        self.args = args
        self.pred_alpha, self.pred_theta = self.initialize_parameters()
    
    def initialize_parameters(self):
        pred_alpha = np.random.uniform(1, 3, self.args.num_samples)
        pred_alpha = Centralize_matrix(pred_alpha)
        pred_theta = truncnorm((-2 - 0) / 1, (2 - 0) / 1, loc=0, scale=1).rvs(size=(self.args.num_samples, self.args.dimension))
        pred_theta = Centralize_matrix(pred_theta)
        
        return pred_alpha, pred_theta

    def train(self, adjacency_matrix):
        # Choose M
        if self.args.type == 'distance1':
            Generate_M = Distance_matrix_M_1
            Gradient_theta = Gradient_theta_binomial_distacne1
            Gradient_alpha = Gradient_alpha_binomial
        elif self.args.type == 'distance2':
            Generate_M = Distance_matrix_M_2
            Gradient_theta = Gradient_theta_binomial_distacne2
            Gradient_alpha = Gradient_alpha_binomial
        elif self.args.type == 'inner-product':
            Generate_M = Inner_matrix_M
            Gradient_theta = Gradient_theta_binomial_inner
            Gradient_alpha = Gradient_alpha_binomial
        else:
            raise ValueError("type must be distacne1, distacne2 or inner-product")
        
        M = Generate_M(self.pred_alpha, self.pred_theta)
        logli = Log_likelihood_binomial(adjacency_matrix, M)
        prev_logli = logli

        flag = True
        iter = 0
        learning_rate = self.args.learning_rate/self.args.num_samples

        while flag:
            M = Generate_M(self.pred_alpha, self.pred_theta)
            grad_y = Gradient_theta(adjacency_matrix, M, self.pred_theta)
            temp_theta = Projection(self.pred_theta + learning_rate * grad_y, self.args.constrain)

            M = Generate_M(self.pred_alpha, temp_theta)
            grad_x = Gradient_alpha(adjacency_matrix, M)
            temp_alpha = Projection(self.pred_alpha + learning_rate * grad_x, self.args.constrain)

            M = Generate_M(temp_alpha, temp_theta)
            temp_logli = Log_likelihood_binomial(adjacency_matrix, M)

            if temp_logli - prev_logli < 0:
                learning_rate /= 5
            else:
                iter += 1
                self.pred_theta = temp_theta
                self.pred_alpha = temp_alpha
                logli = temp_logli

                if self.args.patience == 'relative':
                    if logli - prev_logli > 0 and (logli - prev_logli) / np.abs(prev_logli) < self.args.tolerace:
                        flag = False
                    prev_logli = logli
                elif self.args.patience == 'absolute':
                    if logli - prev_logli > 0 and np.abs(logli - prev_logli) < self.args.tolerace:
                        flag = False
                    prev_logli = logli
                else:
                    raise ValueError("patience must be absolute or relative")
                learning_rate = self.args.learning_rate/self.args.num_samples
        return self.pred_alpha, self.pred_theta, iter
    
    
    
    


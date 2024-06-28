from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import Distance_matrix, Distance_matrix_M_2, Projection, Centralize_matrix
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

class Exp_Parameter_Estimation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Parameter_Estimation, self).__init__(args)
        self.model = self._build_model()
        self.true_alpha, self.true_theta, self.adjacency_matrix = self._get_data()
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self):
        true_alpha, true_theta, adjacency_matrix  = data_provider(self.args)
        return true_alpha, true_theta, adjacency_matrix

    # train
    def train(self):
        start = time.time()
        pred_alpha, pred_theta, iter = self.model.train(self.adjacency_matrix)
        end = time.time()
        elapsed_time = end - start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f'n:{self.args.num_samples}, number:{self.args.number}, iter:{iter}', f'time:{int(hours)}h {int(minutes)}m {seconds:.2f}s')
        
        # create folder
        folder_path =  (f'/home/user/CYH/Code_For_MDS/Project/para_result/{self.args.task_name}/'
                        f'{self.args.type}/{self.args.model}/'
                        f'{self.args.constrain}_{self.args.dimension}_{self.args.learning_rate}'
                        f'{self.args.patience}_{self.args.tolerace}/'
                        f'n_{self.args.num_samples}/{self.args.number}/')

        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # save data
        np.save(folder_path + f'pred_alpha.npy', pred_alpha)
        np.save(folder_path + f'pred_theta.npy', pred_theta)
        np.save(folder_path + f'true_alpha.npy', self.true_alpha)
        np.save(folder_path + f'true_theta.npy', self.true_theta)
        np.save(folder_path + f'adjacency_matrix.npy', self.adjacency_matrix)
        
        return None
        

        

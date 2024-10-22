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
        self.args = args
        
        if args.data == 'Custom':
            self.adjacency_matrix = self._get_data()
        else:
            self.true_alpha, self.true_theta, self.adjacency_matrix = self._get_data()
            
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self):
        result  = data_provider(self.args)
        return result

    # train
    def train(self):
        # create folder
        if self.args.model == 'Normal':
            folder_path =  (f'/home/user/CYH/Code_For_MDS/Project/para_result/{self.args.task_name}/'
                            f'{self.args.type}/{self.args.model}/{self.args.data}'
                            f'{self.args.constrain}_{self.args.dimension}_{self.args.learning_rate}'
                            f'{self.args.patience}_{self.args.tolerace}_{self.args.scale}/'
                            f'n_{self.args.num_samples}/{self.args.seed_number}/')
        else:
            folder_path =  (f'/home/user/CYH/Code_For_MDS/Project/para_result/{self.args.task_name}/'
                f'{self.args.type}/{self.args.model}/{self.args.data}'
                f'{self.args.constrain}_{self.args.dimension}_{self.args.learning_rate}'
                f'{self.args.patience}_{self.args.tolerace}/'
                f'n_{self.args.num_samples}/{self.args.seed_number}/')
        
        # create dir
        os.makedirs(folder_path, exist_ok=True)
        if len(os.listdir(folder_path)) == 0:
            # train
            start = time.time()
            pred_alpha, pred_theta, iter = self.model.train(self.adjacency_matrix) # start training
            end = time.time()
            elapsed_time = end - start
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print(f'n:{self.args.num_samples}, number:{self.args.seed_number}, iter:{iter}', f'time:{int(hours)}h {int(minutes)}m {seconds:.2f}s')
            # save data
            np.save(folder_path + f'pred_alpha.npy', pred_alpha) if self.args.type != 'distance1' else None
            np.save(folder_path + f'pred_theta.npy', pred_theta) 
            np.save(folder_path + f'true_alpha.npy', self.true_alpha) if self.args.type !='distance1' and self.args.data != 'Custom' else None
            np.save(folder_path + f'true_theta.npy', self.true_theta) if self.args.data != 'Custom' else None
            np.save(folder_path + f'adjacency_matrix.npy', self.adjacency_matrix)
        else:
            pass
        
        return None
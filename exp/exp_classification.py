from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import Distance_matrix, Distance_matrix_M_2, Projection, Centralize_matrix
import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# 时间序列预测模型主程序
class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        pass
        
    def train(self, setting):
        pass
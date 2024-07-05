import os
from models import Binomial, Poisson, Normal

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        # 模型字典放在这里
        self.model_dict = {
            'Binomial': Binomial,
            'Poisson': Poisson,
            'Normal': Normal
        }
        self.model = self._build_model()

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def train(self):
        pass


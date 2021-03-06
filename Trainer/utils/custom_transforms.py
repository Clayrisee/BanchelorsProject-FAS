import torchvision.transforms.functional as TF
from random import random

class RandomGammaCorrection:
    def __init__(self, max_gamma, min_gamma):
        self.max_gamma = float(max_gamma)
        self.min_gamma = float(min_gamma)
    
    def __call__(self, x):
        gamma = self.min_gamma + random() * (self.max_gamma - self.min_gamma)
        adjusted_x = TF.adjust_gamma(x, gamma=gamma)
        return adjusted_x
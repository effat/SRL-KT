import os
import torch
import numpy as np


class Saver_reg:
    """Saving pytorch model.
    """
    def __init__(self, savedir, filename, patience=5):
        """
        Arguments:
            savedir (str): Save directory.
            filename (str): Save file name.
            patience (int): How long to wait after last time validation loss improved.
        """
        self.path = os.path.join(savedir, filename)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        self.patience = patience
        ### init with positive inf
        self.score_min = np.Inf
        self.counter = 0

    def save(self, score, network):
        """
        Arguments:
            score (float): Score to maximize.
            network (torch.nn.Module): Network to save if validation loss decreases.
        """

        ### if loss increases, then increse patience. Stop saving model if training loss increases more than patience time
        if score >= self.score_min:
            self.counter += 1
        else:
            self.score_min = score
            torch.save(network, self.path)
            self.counter = 0

        stop = (self.counter >= self.patience)
        return stop

    def load(self):
        return torch.load(self.path)

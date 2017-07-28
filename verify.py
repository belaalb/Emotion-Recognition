from torch.autograd import Variable
import torch

import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
from glob import glob
from tqdm import tqdm


def load_checkpoint(ckpt):

	if os.path.isfile(ckpt):

		ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)

		# Load history
		history = ckpt['history']
		total_iters = ckpt['total_iters']
		cur_epoch = ckpt['cur_epoch']
		its_without_improv = ckpt['its_without_improve']

		plt.plot(history['valid_loss'])
		#plt.plot(history['train_loss'])

		plt.show()

if __name__ == '__main__':

	load_checkpoint('checkpoint_50ep.pt')		


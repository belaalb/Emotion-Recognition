import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

import utils

class DeapDataset(Dataset):

	def __init__(self, root = "/home/isabela/Desktop/emot_recog_class/DEAP_dataset", step = "train"):

		if (step == "train"):
			self.dataset_filename = root + "_train.hdf"
		elif (step == "valid"):
			self.dataset_filename = root + "_valid.hdf"
		else:
			self.dataset_filename = root + "_test.hdf"


	def __len__(self):

		self.length = utils.read_hdf_processed_labels_return_size(self.dataset_filename)

		return self.length

	def __getitem__(self, idx):

		data, label = utils.read_hdf_processed_labels_idx(idx, self.dataset_filename)
		sample = {'data': torch.from_numpy(data).float(), 'label': torch.from_numpy(label).long()}

		return sample




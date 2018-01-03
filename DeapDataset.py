import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

import utils

class DeapDataset(Dataset):

	def __init__(self, root = "/home/isabela/Desktop/emot_recog_class/less_signals_3s/DEAP_dataset", step = "train", model_type = 0, seq_length = 3):

		if (step == "train"):
			self.dataset_filename = root + "_train.hdf"
		elif (step == "valid"):
			self.dataset_filename = root + "_valid.hdf"
		else:
			self.dataset_filename = root + "_test.hdf"

		self.model_type = model_type
		self.seq_length = seq_length


	def __len__(self):

		self.length = utils.read_hdf_processed_labels_return_size(self.dataset_filename) - self.seq_length
		return self.length

	def __getitem__(self, idx):

		if (self.model_type == 0):

			data, label = utils.read_hdf_processed_labels_idx(idx, self.dataset_filename)
			sample = {'data': torch.from_numpy(data).float(), 'label': torch.from_numpy(label).long()}
		
		elif (self.model_type == 1):

			data_seq = []
			label_seq = []			
			for i in range(0, self.seq_length):
				data, label = utils.read_hdf_processed_labels_idx(idx + i, self.dataset_filename)
				data_seq.append(data)
				label_seq.append(label)			

			sample = {'data': torch.from_numpy(data_seq).float(), 'label': torch.from_numpy(label_seq).long()}

		return sample




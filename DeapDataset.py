import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


class DeapDataset(Dataset):

	def __init__(self, root = "DEAP_complete_sequence_", step = "train"):

		if (step == "train"):
			self.dataset_filename = root + "train.hdf"
			self.n_sub = 29		# 29 first subjects for train
			data_key = 'data_s1'	# key for any subject. This is just to get the # of samples per sub and seq length

		elif (step == "valid"):
			self.dataset_filename = root + "valid.hdf"
			self.n_sub = 3		# 3 last subjects from validation
			data_key = 'data_s32'	# key for any subject. This is just to get the # of samples per sub and seq length

		data_file = h5py.File(self.dataset_filename, 'r')
		self.sub_length = data_file[data_key].shape[0]
		self.seq_length = data_file[data_key].shape[1]

		self.length = self.n_sub * self.sub_length - self.seq_length	

		self.step = step	

	def __len__(self):
		
		return self.length

	def __getitem__(self, idx):
			
		data_file = h5py.File(self.dataset_filename, 'r')

		sub = int(np.ceil(idx / self.sub_length))
		
		if (sub == 0):
			sub += 1

		if (self.step == 'valid'):
			sub += 29

		idx = idx % self.sub_length
		data_key = str('data_s' + str(sub))
		labels_key = str('labels_s' + str(sub)) 

		data = data_file[data_key][idx]
		label = data_file[labels_key][idx]

		data_file.close()
				
		sample = {'data': torch.from_numpy(data).float(), 'label': torch.from_numpy(label).long()}		# Only valence label!!

		return sample




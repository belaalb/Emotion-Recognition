import numpy as np
import h5py
import os

import torch



class minibatch_generator:

	def __init__(self, minibatch_size = 32, dataset_filename = "/home/isabela/Desktop/emot_recog_class/DEAP_dataset"):
		
		self.dataset_filename = dataset_filename
		self.dataset_filename_train = self.dataset_filename + "_train.hdf"
		self.dataset_filename_valid = self.dataset_filename + "_valid.hdf"
		self.minibatch_size = minibatch_size


	def minibatch_generator_train(self): #80640 = 32 sub x 60 segments/trial x 40 trials		
	
		open_file = h5py.File(self.dataset_filename_train, 'r')

		i = 0

		while True:

			data_minibatch_train = open_file['data'][i*self.minibatch_size:(i + 1)*self.minibatch_size]
			labels_minibatch_arousal_val_train = open_file['labels_arousal_val'][i*self.minibatch_size:(i + 1)*self.minibatch_size]
   			
			data_minibatch_train = torch.from_numpy(data_minibatch_train)
			data_minibatch_train = data_minibatch_train.float()

			labels_minibatch_arousal_val_train = torch.from_numpy(labels_minibatch_arousal_val_train)
			labels_minibatch_arousal_val_train = labels_minibatch_arousal_val_train.long()

			try:
				current_size = data_minibatch_train.size()[0]
				#print(data_minibatch_train.size())					
				yield (data_minibatch_train, labels_minibatch_arousal_val_train)
				

			except IndexError:
				break
 
			if data_minibatch_train.size()[0] < self.minibatch_size:
				break

			i += 1	

		open_file.close()



	def minibatch_generator_valid(self): #80640 = 32 sub x 60 segments/trial x 40 trials		
	
		open_file = h5py.File(self.dataset_filename_valid, 'r')

		i = 0

		while True:
			
			data_minibatch_valid = open_file['data'][i*self.minibatch_size:(i + 1)*self.minibatch_size]
			labels_minibatch_arousal_val_valid = open_file['labels_arousal_val'][i*self.minibatch_size:(i + 1)*self.minibatch_size]

			data_minibatch_valid = torch.from_numpy(data_minibatch_valid)
			data_minibatch_valid = data_minibatch_valid.float()

			labels_minibatch_arousal_val_valid = torch.from_numpy(labels_minibatch_arousal_val_valid)
			labels_minibatch_arousal_val_valid = labels_minibatch_arousal_val_valid.long()
			
			try:
				current_size = data_minibatch_valid.size()[0]
				yield (data_minibatch_valid, labels_minibatch_arousal_val_valid)

			except IndexError:
				break
 
			if data_minibatch_valid.size()[0] < self.minibatch_size:
				break

			i += 1   			

		open_file.close()

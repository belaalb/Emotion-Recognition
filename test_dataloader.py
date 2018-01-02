import numpy as np
import torch
from DeapDataset import DeapDataset
from torch.utils.data import DataLoader
import utils

deap_dataset = DeapDataset(step='train')
sample = deap_dataset[100]
print(sample['data'].size())
print(sample['label'][0])




# Sampler
#weights = [12422, 22514, 18824] # probability of class 1 = 0.7, of 2 = 0.3 etc

batch_size = 4

'''
weights = 1 / torch.DoubleTensor(weights)
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, 100)

dataloader = DataLoader(deap_dataset, batch_size, shuffle = False, sampler = sampler)

for i_batch, sample_batched in enumerate(dataloader):
	print(i_batch, sample_batched['data'].size(), sample_batched['label'].size())

'''


_, labels_arousal_val = utils.read_hdf_processed_labels('/home/isabela/Desktop/emot_recog_class/DEAP_dataset_train.hdf')

p0 = sum(labels_arousal_val[:, 0] == 0) / labels_arousal_val.shape[0] 	
p1 = sum(labels_arousal_val[:, 0] == 1) / labels_arousal_val.shape[0]
p2 = sum(labels_arousal_val[:, 0] == 2) / labels_arousal_val.shape[0]

probs = [p0, p1, p2]

reciprocal_weights = [0] * len(labels_arousal_val) 

for idx in range(len(labels_arousal_val)):
	reciprocal_weights[idx] = probs[int(labels_arousal_val[idx, 0])]


weights = (1 / torch.Tensor(reciprocal_weights)).double()
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, labels_arousal_val.shape[0])

dataloader = DataLoader(deap_dataset, batch_size, shuffle = False, sampler = sampler)

for i_batch, sample_batched in enumerate(dataloader):
	print(i_batch, sample_batched['data'].size(), sample_batched['label'].size())

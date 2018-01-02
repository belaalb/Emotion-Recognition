from torch.autograd import Variable
import torch

import numpy as np
import matplotlib.pyplot as plt
import pickle

import os
from glob import glob
from tqdm import tqdm
import utils
from torch.utils.data import DataLoader
from DeapDataset import DeapDataset
import model_arousal_eeg_gsr_temp_convtemporal

from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, roc_auc_score


def load_checkpoint(ckpt):

	if os.path.isfile(ckpt):

		ckpt = torch.load(ckpt, map_location = lambda storage, loc: storage)		

	else:
		print('No checkpoint found at {}'.format(ckpt)) 
		ckpt = None

	return ckpt


def plot_learningcurves(history, *keys):

	for key in keys:
		plt.plot(history[key])
	
	plt.show()


def calculate_metrics(labels, output, out_dict):

	out_fusion_max = np.argmax(output, axis = 1)


	acc = accuracy_score(labels, out_fusion_max)
	out_dict['acc'].append(acc)

	precision = precision_score(labels, out_fusion_max)
	out_dict['precision'].append(precision)

	recall = recall_score(labels, out_fusion_max)
	out_dict['recall'].append(recall)

	f1 = f1_score(labels, out_fusion_max)
	out_dict['f1'].append(f1)

	auc = roc_auc_score(labels, out_fusion_max)
	out_dict['auc'].append(auc)

	return out_dict

	


def test_model_classification_decision_level(model, ckpt, dataloader_valid, cuda_mode):

	model.load_state_dict(ckpt['model_state'])

	out = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

	for t, batch in enumerate(dataloader_valid):

		model.eval()

		x = batch['data'] 
		y = batch['label']

		if cuda_mode:
			x = x.cuda()
			y = y.cuda()
			model = model.cuda()

		x = Variable(x, requires_grad = False)
		y = Variable(y, requires_grad = False)

		out_eeg, out_temp_gsr = model.forward_multimodal_arousal(x)

		out_fusion = (out_eeg + out_temp_gsr) / 2

		
		labels = y[:, 1].cpu().data.numpy()

		metrics = calculate_metrics(labels, out_fusion.cpu().data.numpy(), out)


	return metrics




if __name__ == '__main__':

	reciprocal_weights_valid, length_valid = utils.calculate_weights(step = 'valid')
	weight_valid = 1 / torch.DoubleTensor(reciprocal_weights_valid)
	dataset_valid = DeapDataset(step = 'valid')
	sampler_valid = torch.utils.data.sampler.WeightedRandomSampler(weight_valid, length_valid)
	dataloader_valid = DataLoader(dataset_valid, 400)
	cuda_mode = True

	model = model_arousal_eeg_gsr_temp_convtemporal.model()
	ckpt = '/home/isabela/Desktop/emot_recog_class/less_signals_3s/decision_level/valence_last_trial/checkpoint_15ep.pt'

	ckpt = load_checkpoint(ckpt)
	
	metrics = test_model_classification_decision_level(model, ckpt, dataloader_valid, cuda_mode)

	#history = ckpt['history']
	#plot_learningcurve(history, 'train_loss', 'valid_loss')

	for key in metrics.keys():
		print(key)		
		print(sum(metrics[key])/len(metrics[key]))

	


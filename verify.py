from torch.autograd import Variable
import torch

import numpy as np
import matplotlib.pyplot as plt


import os
from torch.utils.data import DataLoader
from DeapDataset import DeapDataset
import model_tempconv_lstm

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

	


def test_model_classification(model, ckpt, dataloader_valid, cuda_mode):

	model.load_state_dict(ckpt['model_state'])

	out_metrics = {'acc': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

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

		out = model.forward(x)

		
		labels = y[:, 0].cpu().data.numpy()

		metrics = calculate_metrics(labels, out.cpu().data.numpy(), out_metrics)


	return metrics




if __name__ == '__main__':


	dataset_valid = DeapDataset(step = 'valid')
	dataloader_valid = DataLoader(dataset_valid, 50)
	cuda_mode = True

	model = model_tempconv_lstm.model_eeg_short()
	ckpt = 'checkpoint_4ep.pt'

	ckpt = load_checkpoint(ckpt)
	
	#metrics = test_model_classification(model, ckpt, dataloader_valid, cuda_mode)

	history = ckpt['history']
	plot_learningcurves(history, 'train_loss', 'valid_loss')

	#for key in metrics.keys():
	#	print(key)		
	#	print(sum(metrics[key])/len(metrics[key]))

	


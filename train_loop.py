from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
from DeapDataset import DeapDataset
from torch.utils.data import DataLoader

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

import gc

from sklearn.metrics import precision_score, f1_score, recall_score


class TrainLoop(object):

	def __init__(self, model, optimizer, minibatch_size, checkpoint_path = None, checkpoint_epoch = None, cuda = True):
		
		if checkpoint_path is None:

			# Save to current directory
			self.checkpoint_path = os.getcwd()

		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):

				os.mkdir(self.checkpoint_path)
	
		self.save_every_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}it.pt')
		self.save_epoch_fmt = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.cuda_mode = cuda


		if checkpoint_epoch is None:
			self.model = model
			self.optimizer = optimizer
			self.minibatch_size = minibatch_size
			self.history = {'train_loss': [], 'valid_loss': []}
			self.total_iters = 0
			self.cur_epoch = 0
			self.its_without_improv = 0
			self.last_best_val_loss = np.inf
			self.initialize_params()

		else:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))


		self.dataset_train = DeapDataset(step = 'train')
		self.dataloader_train = DataLoader(self.dataset_train, self.minibatch_size, shuffle = True)

		self.dataset_valid = DeapDataset(step = 'valid')
		self.dataloader_valid = DataLoader(self.dataset_valid, self.minibatch_size)


	def train(self, n_epochs = 1, patience = 5):
		# Note: Logging expects the losses to be divided by the batch size

		last_val_loss = float('inf')

		while (self.cur_epoch < n_epochs) and (self.its_without_improv < patience):
			train_loss_sum = 0.0
			train_accuracy = 0.0
			self.total_iters = 0.0
			

			print('Epoch {}/{}'.format(self.cur_epoch + 1, n_epochs))
			train_iter = tqdm(enumerate(self.dataloader_train))

			for it, batch in train_iter:
				loss, acc = self.train_step(batch, it)
	
				train_loss_sum += loss
				self.total_iters += 1

				train_accuracy += acc

			train_loss_avg = train_loss_sum / (it + 1)  	
			train_accuracy_avg = train_accuracy / (it + 1) 

			self.history['train_loss'].append(train_loss_avg)
 
			print('Training loss: {}'.format(train_loss_avg))			
			print('Training accuracy: {}'.format(train_accuracy_avg))


			# Validation
			valid_loss_sum = 0.0
			valid_accuracy = 0.0

			for it, batch in enumerate(self.dataloader_valid):
				
				loss, acc = self.valid(batch)

				valid_loss_sum += loss
				valid_accuracy += acc	

			valid_loss_avg = valid_loss_sum / (it + 1)
			valid_accuracy_avg = valid_accuracy / (it + 1)

			self.history['valid_loss'].append(valid_loss_avg)

			print('Validation loss: {}'.format(valid_loss_avg)) 	
			print('Validation accuracy: {}'.format(valid_accuracy_avg))


			self.cur_epoch += 1

			if self.history['valid_loss'][-1] < self.last_best_val_loss:
				self.last_best_val_loss = self.history['valid_loss'][-1]
				self.its_without_improv = 0
				self.checkpointing()
			else:
				self.its_without_improv += 1

			if self.its_without_improv > patience:
				self.its_without_improv = 0
				self.update_lr()


		# saving final models
		print('Saving final model...')

		torch.save(self.model.state_dict(), './model1.pt')

	def train_step(self, batch, curr_iter):

		self.model.train()
			
		x = batch['data']
		y = batch['label']
		
		x = Variable(x)
		y = Variable(y, requires_grad = False)

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		self.optimizer.zero_grad()

		out = self.model.forward(x)

		targets = y[:, 0].contiguous()
		targets = targets.view(targets.size(0), 1)
		targets = targets.float()

		loss_calc = F.binary_cross_entropy(out, targets) + F.mse_loss(out.mean(), targets.mean()) # + F.mse_loss(out.std(), targets.std())

		loss_calc.backward()
		self.optimizer.step()

		loss_return = torch.sum(loss_calc.data)

		class_pred = out.data.gt(0.5 * torch.ones_like(out.data)).float()

		correct = class_pred.eq(targets.data).sum()
		accuracy_return = 100.0 * correct / len(y)

		self.check_nans()

		print('\n')

		self.print_params_norm()

		self.print_grad_norm()

		if (curr_iter % 10 == 0):

			#class_pred = class_pred.cpu().numpy()
			#targets = targets.cpu().data.numpy()

			#print('Precision')
			#print(precision_score(targets, class_pred))
			#print('Recall')
			#print(recall_score(targets, class_pred))
			#print('F1 score')
			#print(f1_score(targets, class_pred))

			print('Accuracy on minibatch:', accuracy_return)
			
			# Number of predicted 1s and 0s
			pred_ones = class_pred.eq(torch.ones_like(class_pred)).sum()
			pred_zeros = class_pred.eq(torch.zeros_like(class_pred)).sum()

			print('Predicted ones:', pred_ones)
			print('Predicted zeros:', pred_zeros)
		
 	

		return loss_return, accuracy_return


	def valid(self, batch):

		self.model.eval()
		
		x = batch['data']
		y = batch['label']

		x = Variable(x, requires_grad = False)
		y = Variable(y, requires_grad = False)

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()
			

		out = self.model.forward(x)

		targets = y[:, 0].contiguous()
		targets = targets.view(targets.size(0), 1)
		targets = targets.float()

		loss_return = F.binary_cross_entropy(out, targets) + F.mse_loss(out.mean(), targets.mean()) # + F.mse_loss(out.std(), targets.std())

		loss_return = loss_return.data[0]

		class_pred = out.data.gt(0.5 * torch.ones_like(out.data)).float()

		correct = class_pred.eq(targets.data).sum()
		accuracy_return = 100.0 * correct / len(y)


		return loss_return, accuracy_return

	def checkpointing(self):
		
		# Checkpointing
		print('Checkpointing...')
		ckpt = {'model_state': self.model.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch,
		'its_without_improve': self.its_without_improv}
		torch.save(ckpt, self.save_epoch_fmt.format(self.cur_epoch))

	def update_lr(self):

		for param_group in self.optimizer.param_groups:
			param_group['lr'] = max(param_group['lr']/10., 0.000001)
		print('updating lr to: {}'.format(param_group['lr']))

	def initialize_params(self):

		for layer in self.model.modules():
			if isinstance(layer, torch.nn.Conv1d):
		  		nn.init.kaiming_normal(layer.weight.data)


	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			
			# Load model state
			self.model.load_state_dict(ckpt['model_state'])
			
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']
			self.its_without_improv = ckpt['its_without_improve']

		else:

			print('No checkpoint found at {}'.format(ckpt))

	# Debugging stuff

	def check_nans(self):

		for layer in self.model.modules():	
			for params in list(layer.parameters()):
				if np.any(np.isnan(params.data.cpu().numpy())):
					print('params NANs!!!!!')
				if np.any(np.isnan(params.grad.data.cpu().numpy())):
					print('grads NANs!!!!!!')			


	def print_params_norm(self):

		norm = 0.0
	
		for params in list(self.model.parameters()):
		
			norm += params.norm(2).data[0]

		print('Sum of weights norms: {}'.format(norm))


	def print_grad_norm(self):

		norm = 0.0

		for i, params in enumerate(list(self.model.parameters())):
			
			norm += params.grad.norm(2).data[0]

		print('Sum of grads norms: {}'.format(norm))



     

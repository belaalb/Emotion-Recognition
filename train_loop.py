from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.nn as nn
from DeapDataset import DeapDataset
from torch.utils.data import DataLoader
import utils

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

import gc

from sklearn.metrics import precision_score, f1_score, recall_score


class TrainLoop(object):

	def __init__(self, model, optimizer, minibatch_size, model_type = 0, seq_length = 3, checkpoint_path = None, checkpoint_epoch = None, cuda = True):
		
		if checkpoint_path is None:

			# Save to current directory
			self.checkpoint_path = os.getcwd()

		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):

				os.mkdir(self.checkpoint_path)
	
		self.model_type = model_type
		self.seq_length = seq_length
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

		else:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))


		reciprocal_weights_train, length_train = utils.calculate_weights(step = 'train')
		self.weight_train = 1 / torch.DoubleTensor(reciprocal_weights_train)
		self.dataset_train = DeapDataset(step = 'train', model_type = self.model_type, seq_length = self.seq_length)
		self.sampler_train = torch.utils.data.sampler.WeightedRandomSampler(self.weight_train, length_train)
		self.dataloader_train = DataLoader(self.dataset_train, self.minibatch_size, shuffle = False, sampler = self.sampler_train)


		reciprocal_weights_valid, length_valid = utils.calculate_weights(step = 'valid')

		self.weight_valid = 1 / torch.DoubleTensor(reciprocal_weights_valid)
		self.dataset_valid = DeapDataset(step = 'valid', model_type = self.model_type, seq_length = self.seq_length)
		self.sampler_valid = torch.utils.data.sampler.WeightedRandomSampler(self.weight_valid, length_valid)
		self.dataloader_valid = DataLoader(self.dataset_valid, self.minibatch_size)





	def train(self, n_epochs = 1, patience = 100, n_workers = 1, tl_delay = 1, save_every = None):
		# Note: Logging expects the losses to be divided by the batch size

		last_val_loss = float('inf')


		while (self.cur_epoch < n_epochs) and (self.its_without_improv < patience):
			train_loss_sum = 0.0
			train_accuracy = 0.0
			self.iter_epoch = 0.0
			

			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.dataloader_train))

			print(train_iter)

			print(self.cur_epoch)

			for t, batch in train_iter:

				loss, acc = self.train_step(batch)

				self.history['train_loss'].append(loss)	
				
				train_loss_sum += loss
				self.total_iters += 1
				self.iter_epoch += 1
				
				train_accuracy += acc

				print(acc)
				
				if save_every is not None:
					if self.total_iters % save_every == 0:
						torch.save(self, self.save_every_fmt.format(self.total_iters))

				
			train_accuracy_avg = train_accuracy / self.iter_epoch  
			print('Training accuracy: {}'.format(train_accuracy_avg))


			# Validation
			valid_loss_sum = 0.0
			n_valid_iterations = 0
			valid_accuracy = 0.0

			for t, batch in enumerate(self.dataloader_valid):
				
				loss, acc = self.valid(batch)
				
				self.history['valid_loss'].append(loss)

				valid_loss_sum += loss
				n_valid_iterations += 1

				valid_accuracy += acc	

				print(acc)


			valid_accuracy_avg = valid_accuracy / n_valid_iterations 	
			print('Validation accuracy: {}'.format(valid_accuracy_avg))

			valid_loss_avg = valid_loss_sum / n_valid_iterations

			self.checkpointing()

			self.cur_epoch += 1

			if valid_loss_avg < last_val_loss:
				self.its_without_improv = 0

			else:
				self.its_without_improv += 1

			last_val_loss = valid_loss_avg	

		# saving final models
		print('Saving final model...')

		torch.save(self.model.state_dict(), './model1.pt')

	def train_step(self, batch):

		self.model.train()
			
		x = batch['data']
		y = batch['label']
		
		print('\n')

		x = Variable(x)
		y = Variable(y, requires_grad = False)

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		self.optimizer.zero_grad()

		out = self.model.forward(x)

		targets = y[:, 0].contiguous()
		targets = targets.view(targets.size(0), 1)

		loss_calc = F.binary_cross_entropy(out, y[:, 0].float())

		loss_calc.backward()
		self.optimizer.step()

		loss_return = torch.sum(loss_calc.data)
		
		out_max = (torch.max(out, 1)[1])

		accuracy = torch.mean((out_max == y[:, 0]).float())
		accuracy_return = accuracy.data


		if (self.iter_epoch % 500 == 0):

			out_max = out_max.cpu().data.numpy()
			targets = targets.cpu().data.numpy()

			print('Precision')
			print(precision_score(targets, out_max))
			print('Recall')
			print(recall_score(targets, out_max))
			print('F1 score')
			print(f1_score(targets, out_max))
			print(out_max)
 	

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

		loss_calc = F.binary_cross_entropy(out, y[:, 0].float())

		out_max = (torch.max(out, 1)[1])


		accuracy = torch.mean((out_max == y[:, 0]).float())
		accuracy_return = accuracy.data

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

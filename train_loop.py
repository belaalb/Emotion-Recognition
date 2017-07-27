from torch.autograd import Variable
import torch

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, model, optimizer, generator, checkpoint_path = None, checkpoint_epoch = None, cuda = True):
		
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
			self.generator = generator
			self.history = {'train_loss': [], 'valid_loss': []}
			self.total_iters = 0
			self.cur_epoch = 0
			self.its_without_improv = 0

		else:
			self.load_checkpoint(self.save_epoch_fmt.format(checkpoint_epoch))


	def train(self, n_epochs = 1, patience = 100, n_workers = 1, tl_delay = 1, save_every = None):
		# Note: Logging expects the losses to be divided by the batch size

		last_val_loss = float('inf')

		while (self.cur_epoch < n_epochs) and (self.its_without_improv < patience):
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.generator.minibatch_generator_train()))

			print(self.cur_epoch)

			for t, batch in train_iter:

				train_loss = self.train_step(batch)
				self.history['train_loss'].append(train_loss)
				self.total_iters += 1
				if save_every is not None:
					if self.total_iters % save_every == 0:
						torch.save(self, self.save_every_fmt.format(self.total_iters))

			# Validation
			val_loss = 0.0
			n_val_samples = 0

			for t, batch in enumerate(self.generator.minibatch_generator_valid()):
				loss = self.valid(batch)
				val_loss = val_loss + loss
				n_val_samples += batch[0].size()[0]

			val_loss /= n_val_samples	
			
			print('Validation loss: {}'.format(val_loss))
			
			self.history['valid_loss'].append(val_loss)

			self.checkpointing()

			self.cur_epoch += 1

			if val_loss < last_val_loss:
				self.its_without_improv = 0

			else:
				self.its_without_improv += 1

			last_val_loss = val_loss	

		# saving final models
		print('Saving final model...')

		torch.save(self.model.state_dict(), './model1.pt')

	def train_step(self, batch):

		self.model.train()
		
		x, y = batch
		print('\n')
		print(x.size())
		print(y.size())

		x = Variable(x)
		y = Variable(y, requires_grad = False)

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		out = self.model.forward(x)

		loss = torch.mean(torch.nn.functional.pairwise_distance(out, y))

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

		return loss.data[0]
		
	def valid(self, batch):

		self.model.eval()
		
		x, y = batch
		x = Variable(x, requires_grad = False)
		y = Variable(y, requires_grad = False)

		if self.cuda_mode:
			x = x.cuda()
			y = y.cuda()

		out = self.model.forward(x)
		loss = torch.nn.functional.pairwise_distance(out, y)			#checar

		loss_return = torch.sum(loss.data)							# If sum receives: i) Tensor (loss.data), it returns a float; ii) Variable (loss), it returns a tensor

		return loss_return

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

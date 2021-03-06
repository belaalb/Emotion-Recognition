import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

'''
Model 1: Temporal convolution + lstm + feature fusion, for valence classification
- EEG + GSR and skin temperature
- 3 seconds long samples
'''

class model_multimodal(nn.Module):
	
	def __init__(self):

		super(model_multimodal, self).__init__()

		self.features_eeg = nn.Sequential(
			nn.Conv1d(32, 48, kernel_size = 128, bias = False),		# 384 - 128 + 1 = 257
			nn.BatchNorm1d(48),
			nn.ReLU(),
			nn.Conv1d(48, 64, kernel_size = 128, bias = False),		# 257 - 128 + 1 = 130
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 64, kernel_size = 64, bias = False),		# 130 - 64 + 1 = 67
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 128, kernel_size = 16, bias = False),		# 67 - 16 + 1 = 52
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 128, kernel_size = 16, bias = False),		# 52 - 16 + 1 = 37
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 256, kernel_size = 16, bias = False),		# 37 - 16 + 1 = 22
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Conv1d(256, 256, kernel_size = 8, bias = False),		# 22 - 8 + 1 = 15
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Conv1d(256, 256, kernel_size = 8, bias = False),		# 15 - 8 + 1 = 8
			nn.BatchNorm1d(256),
			nn.ReLU())

		self.features_eeg_flatten = nn.Sequential(			
			nn.Linear(256*8, 1024, bias = False),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 512, bias = False),
			nn.BatchNorm1d(512),
			nn.ReLU(),			
			nn.Linear(512, 128))				# Representation to be concatenated
		
		self.features_others = nn.Sequential(
			nn.Conv1d(2, 16, kernel_size = 128, bias = False),	# 384 - 192 + 1 = 257
			nn.BatchNorm1d(16),
			nn.ReLU(),			
			nn.Conv1d(16, 32, kernel_size = 128, bias = False),	# 257 - 128 + 1 = 130
			nn.BatchNorm1d(32),
			nn.ReLU(),			
			nn.Conv1d(32, 64, kernel_size = 64, bias = False),	# 130 - 64 + 1 = 67
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 64, kernel_size = 32, bias = False),	# 67 - 32 + 1 = 36
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 128, kernel_size = 16, bias = False),	# 36 - 16 + 1 = 21
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 128, kernel_size = 8, bias = False),	# 21 - 8 + 1 = 14
			nn.BatchNorm1d(128),
			nn.ReLU())

		self.features_others_flatten = nn.Sequential(			
			nn.Linear(128*14, 512, bias = False),
			nn.BatchNorm1d(512),
			nn.ReLU(),			
			nn.Linear(512, 128))				# Representation to be concatenated

		self.n_hidden_layers = 1
		self.hidden_size = 64
		self.lstm = nn.LSTM(128 + 128, self.hidden_size, self.n_hidden_layers, bidirectional = False)
		self.fc_lstm = nn.Linear(self.hidden_size, 8)

		# Output layer
		self.fc_out = nn.Linear(8, 1)
			

	def forward(self, x):

		minibatch_size = x.size(0)
		seq_length = x.size(1)
	
		x_eeg = x[:, :, 0:32, :]
		x_eeg = x_eeg.contiguous()
		x_eeg = x_eeg.view(minibatch_size*seq_length, x_eeg.size(-2), x_eeg.size(-1))

		x_gsr = x[:, :, 36, :]
		x_temp = x[:, :, 39, :]
		x_gsr = x_gsr.contiguous()
		x_temp = x_temp.contiguous()
		x_gsr = x_gsr.view(x_gsr.size(0), x_gsr.size(1) , 1,  x_gsr.size(-1))
		x_temp = x_temp.view(x_temp.size(0), x_temp.size(1), 1,  x_temp.size(-1))
		x_others = torch.cat([x_temp, x_gsr], 2)
		x_others = x_others.view(minibatch_size*seq_length, x_others.size(-2), x_others.size(-1))

		#EEG
		x_eeg = self.features_eeg(x_eeg)
		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		x_eeg = self.features_eeg_flatten(x_eeg)

		#Skin temp and GSR
		x_others  = self.features_others(x_others)
		x_others = x_others.view(x_others.size(0), -1)

		x_others = self.features_others_flatten(x_others)
		
		concatenated_output = torch.cat([x_eeg, x_others], 1)

		# Reshape lstm input 
		concatenated_output = concatenated_output.view(seq_length, minibatch_size, concatenated_output.size(-1))

		# Initial hidden states
		h0 = Variable(torch.zeros(self.n_hidden_layers, concatenated_output.size(1), self.hidden_size))  
		c0 = Variable(torch.zeros(self.n_hidden_layers, concatenated_output.size(1), self.hidden_size))

		if x.is_cuda:
			h0 = h0.cuda()
			c0 = c0.cuda()	

		seq_out, _ = self.lstm(concatenated_output, (h0, c0))

		output = self.fc_lstm(seq_out[-1])
		output = F.relu(output)
		output = F.softmax(self.fc_out(output), 0)

		return output
	
'''
Model 2: Temporal convolution + lstm for valence classification
- Only EEG 
- 1 second long samples
'''

class model_eeg_rnn(nn.Module):
	
	def __init__(self):

		super(model_eeg_rnn, self).__init__()

		self.features_eeg = nn.Sequential(
			nn.Conv1d(32, 48, kernel_size = 32, bias = False),		# 128 - 32 + 1 = 97
			nn.BatchNorm1d(48),
			nn.ReLU(),
			nn.Conv1d(48, 64, kernel_size = 32, bias = False),		# 97 - 32 + 1 = 66
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 64, kernel_size = 16, bias = False),		# 66 - 16 + 1 = 51
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 128, kernel_size = 16, bias = False),		# 51 - 16 + 1 = 36
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 128, kernel_size = 16, bias = False),		# 36 - 16 + 1 = 21
			nn.BatchNorm1d(128),
			nn.ReLU(),
			nn.Conv1d(128, 256, kernel_size = 8, bias = False),		# 21 - 8 + 1 = 14
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Conv1d(256, 256, kernel_size = 4, bias = False),		# 14 - 4 + 1 = 11
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Conv1d(256, 256, kernel_size = 4, bias = False),		# 11 - 4 + 1 = 8
			nn.BatchNorm1d(256),
			nn.ReLU())

		self.features_eeg_flatten = nn.Sequential(			
			nn.Linear(256*8, 1024, bias = False),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 512, bias = False),
			nn.BatchNorm1d(512),
			nn.ReLU(),			
			nn.Linear(512, 128))				# Representation to be concatenated
		
		self.n_hidden_layers = 2
		self.hidden_size = 80
		self.lstm = nn.LSTM(128, self.hidden_size, self.n_hidden_layers, bidirectional = False)
		self.fc_lstm = nn.Linear(self.hidden_size, 40)

		# Output layer
		self.fc_out = nn.Linear(40, 1)
			

	def forward(self, x):

		minibatch_size = x.size(0)
		seq_length = x.size(1)
	
		x_eeg = x[:, :, 0:32, :]
		x_eeg = x_eeg.contiguous()
		x_eeg = x_eeg.view(minibatch_size*seq_length, x_eeg.size(-2), x_eeg.size(-1))

		#EEG
		x_eeg = self.features_eeg(x_eeg)
		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		x_eeg = self.features_eeg_flatten(x_eeg)

		
		# Reshape lstm input 
		lstm_input = x_eeg.view(seq_length, minibatch_size, x_eeg.size(-1))

		# Initial hidden states
		h0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))  
		c0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))

		if x.is_cuda:
			h0 = h0.cuda()
			c0 = c0.cuda()	

		seq_out, _ = self.lstm(lstm_input, (h0, c0))

		output = self.fc_lstm(seq_out[-1])
		output = F.relu(output)
		output = F.sigmoid(self.fc_out(output))

		return output
		

'''
Model 3: Shorter version of temporal convolution + lstm for valence classification
- Only EEG 
- 1 second long samples
'''

class model_eeg_rnn_short(nn.Module):
	
	def __init__(self):

		super(model_eeg_rnn_short, self).__init__()

		self.features_eeg = nn.Sequential(
			nn.Conv1d(32, 48, kernel_size = 32),		# 128 - 32 + 1 = 97
			nn.ReLU(),
			nn.MaxPool1d(16, stride = 1),				# Out = 82

			nn.Conv1d(48, 64, kernel_size = 16),		# 82 - 32 + 1 = 67
			nn.ReLU(),
			nn.MaxPool1d(8, stride = 1),				# Out = 60

			nn.Conv1d(64, 64, kernel_size = 16),		# 60 - 16 + 1 = 45
			nn.ReLU(),
			nn.MaxPool1d(8, stride = 1),				# Out = 38
			
			nn.Conv1d(64, 64, kernel_size = 16),		# 38 - 16 + 1 = 23
			nn.ReLU(),
			nn.MaxPool1d(4, stride = 1),				# Out = 20
			
			nn.Conv1d(64, 100, kernel_size = 8),		# 20 - 8 + 1 = 13
			nn.ReLU(),
			nn.AvgPool1d(4, stride = 1))				# Out = 10


		self.features_eeg_flatten = nn.Sequential(			
			nn.Linear(100*10, 500),
			nn.ReLU(),		
			nn.Linear(500, 128),				# Representation to be concatenated
			nn.ReLU())		

		self.n_hidden_layers = 1
		self.hidden_size = 80
		self.lstm = nn.LSTM(128, self.hidden_size, self.n_hidden_layers, bidirectional = False)
		self.fc_lstm = nn.Linear(self.hidden_size, 1)

		# Output layer
		#self.fc_out = nn.Linear(40, 1)


	def forward(self, x):

		minibatch_size = x.size(0)
		seq_length = x.size(1)
	
		x_eeg = x[:, :, 0:32, :]
		x_eeg = x_eeg.contiguous()
		x_eeg = x_eeg.view(minibatch_size*seq_length, x_eeg.size(-2), x_eeg.size(-1))

		#EEG
		x_eeg = self.features_eeg(x_eeg)
		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		x_eeg = self.features_eeg_flatten(x_eeg)

		
		# Reshape lstm input 
		lstm_input = x_eeg.view(seq_length, minibatch_size, x_eeg.size(-1))

		# Initial hidden states
		h0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))  
		c0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))

		if x.is_cuda:
			h0 = h0.cuda()
			c0 = c0.cuda()	

		seq_out, _ = self.lstm(lstm_input, (h0, c0))

		output = self.fc_lstm(seq_out[-1])
		#output = F.relu(output)
		#output = F.sigmoid(self.fc_out(output))
		output = F.hardtanh(output, 0, 1)

		return output
		
'''
Model 4: features extractor architecture based on "Deep Learning With Convolutional Neural
Networks for EEG Decoding and Visualization"
- Only EEG 
- 1 second long samples
'''

class model_2Dconv(nn.Module):
	
	def __init__(self):

		super(model_2Dconv, self).__init__()

		self.features_eeg = nn.Sequential(
			nn.Conv2d(1, 25, kernel_size = (10, 1)),		# 128 - 10 + 1 = 119

			nn.Conv2d(25, 1, kernel_size = (1, 32)),		# 82 - 32 + 1 = 67
			nn.ELU(),
			nn.MaxPool2d((3, 1), stride = 1),				# Out = 60

			#RESHAPE!! x = x.view(-1, 1, 25, 119)

			nn.Conv2d(1, 1, kernel_size = (10, 25)),		    # 60 - 16 + 1 = 45
			nn.ReLU(),
			nn.MaxPool2d((3, 1), stride = 1),				# Out = 38
			
			nn.Conv2d(64, 64, kernel_size = 16),		# 38 - 16 + 1 = 23
			nn.ReLU(),
			nn.MaxPool2d(4, stride = 1),				# Out = 20
			
			nn.Conv2d(64, 100, kernel_size = 8),		# 20 - 8 + 1 = 13
			nn.ReLU(),
			nn.AvgPool2d(4, stride = 1))				# Out = 10


		self.features_eeg_flatten = nn.Sequential(			
			nn.Linear(100*10, 500),
			nn.ReLU(),		
			nn.Linear(500, 128),				# Representation to be concatenated
			nn.ReLU())		

		self.n_hidden_layers = 1
		self.hidden_size = 80
		self.lstm = nn.LSTM(128, self.hidden_size, self.n_hidden_layers, bidirectional = False)
		self.fc_lstm = nn.Linear(self.hidden_size, 1)

		# Output layer
		#self.fc_out = nn.Linear(40, 1)


	def forward(self, x):

		minibatch_size = x.size(0)
		seq_length = x.size(1)
	
		x_eeg = x[:, :, 0:32, :]
		x_eeg = x_eeg.contiguous()
		x_eeg = x_eeg.view(minibatch_size*seq_length, x_eeg.size(-2), x_eeg.size(-1))

		#EEG
		x_eeg = self.features_eeg(x_eeg)
		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		x_eeg = self.features_eeg_flatten(x_eeg)

		
		# Reshape lstm input 
		lstm_input = x_eeg.view(seq_length, minibatch_size, x_eeg.size(-1))

		# Initial hidden states
		h0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))  
		c0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))

		if x.is_cuda:
			h0 = h0.cuda()
			c0 = c0.cuda()	

		seq_out, _ = self.lstm(lstm_input, (h0, c0))

		output = self.fc_lstm(seq_out[-1])
		#output = F.relu(output)
		#output = F.sigmoid(self.fc_out(output))
		output = F.hardtanh(output, 0, 1)

		return output


'''
Model 5: features extractor architecture based on ChronoNet
- Only EEG 
- 3 second long samples
- Next steps: implement concatenated convolutions
'''

class model_eeg_rnn_shorterconvs(nn.Module):
	
	def __init__(self):

		super(model_eeg_rnn_shorterconvs, self).__init__()

		self.features_eeg = nn.Sequential(
			nn.Conv1d(32, 64, kernel_size = 4),		# 384 - 4 + 1 = 381
			nn.ReLU(),

			nn.Conv1d(64, 64, kernel_size = 4),		# 381 - 4 + 1 = 378
			nn.ReLU(),

			nn.Conv1d(64, 64, kernel_size = 4),		# 378 - 4 + 1 = 375 
			nn.ReLU())
				

		self.n_hidden_layers = 4
		self.hidden_size = 64
		self.lstm = nn.LSTM(64, self.hidden_size, self.n_hidden_layers, bidirectional = False)
		self.fc_lstm = nn.Linear(self.hidden_size, 1)

		# Output layer
		#self.fc_out = nn.Linear(40, 1)


	def forward(self, x):

		minibatch_size = x.size(0)
	
		x_eeg = x[:, 0:32, :]
		x_eeg = x_eeg.contiguous()

		#EEG
		x_eeg = self.features_eeg(x_eeg)
		
		# Reshape lstm input 
		lstm_input = x_eeg.view(x_eeg.size(-1), minibatch_size, x_eeg.size(-2))

		# Initial hidden states
		h0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))  
		c0 = Variable(torch.zeros(self.n_hidden_layers, lstm_input.size(1), self.hidden_size))

		if x.is_cuda:
			h0 = h0.cuda()
			c0 = c0.cuda()	

		seq_out, _ = self.lstm(lstm_input, (h0, c0))

		output = self.fc_lstm(seq_out[-1])
		output = F.sigmoid(output)


		return output
				

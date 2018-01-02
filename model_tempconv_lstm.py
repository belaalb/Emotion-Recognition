import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Model 1: Temporal convolution + lstm + feature fusion, for arousal classification

class model(nn.Module):
	
	def __init__(self):

		super(model, self).__init__()

		self.features_eeg = nn.Sequential(
			nn.Conv1d(32, 48, kernel_size = 128),		# 384 - 128 + 1 = 257
			nn.BatchNorm1d(48),
			nn.ReLU(),
			nn.Conv1d(48, 64, kernel_size = 128),		# 257 - 128 + 1 = 130
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 64, kernel_size = 64),		# 130 - 64 + 1 = 67
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Conv1d(64, 32, kernel_size = 16),		# 67 - 16 + 1 = 52
			nn.BatchNorm1d(32),
			nn.ReLU(),
			nn.Conv1d(32, 16, kernel_size = 16),		# 52 - 16 + 1 = 37
			nn.BatchNorm1d(16),
			nn.ReLU(),
			nn.Conv1d(16, 8, kernel_size = 16),		# 37 - 16 + 1 = 22
			nn.BatchNorm1d(8),
			nn.ReLU(),
			nn.Conv1d(8, 4, kernel_size = 8),		# 22 - 8 + 1 = 15
			nn.BatchNorm1d(4),
			nn.ReLU(),
			nn.Conv1d(4, 4, kernel_size = 8),		# 15 - 8 + 1 = 8
			nn.BatchNorm1d(4),
			nn.ReLU())

		self.features_eeg_flatten = nn.Sequential(			
			nn.Linear(4*8, 16),
			nn.BatchNorm1d(16),
			nn.ReLU(),			
			nn.Linear(16, 8))				# Representation to be concatenated
		
		self.features_others = nn.Sequential(
			nn.Conv1d(2, 16, kernel_size = 192),	# 384 - 192 + 1 = 193
			nn.BatchNorm1d(16),
			nn.ReLU(),			
			nn.Conv1d(16, 32, kernel_size = 128),	# 193 - 128 + 1 = 66
			nn.BatchNorm1d(32),
			nn.ReLU(),			
			nn.Conv1d(32, 16, kernel_size = 64),	# 66 - 64 + 1 = 3
			nn.BatchNorm1d(16),
			nn.ReLU())

		self.features_others_flatten = nn.Sequential(			
			nn.Linear(16*3, 16),
			nn.BatchNorm1d(16),
			nn.ReLU(),			
			nn.Linear(16, 8))				# Representation to be concatenated

		self.lstm = nn.LSTM(16, 4, 2, bidirectional = True)
		self.fc_lstm = nn.Linear(4*2, 4)

		# Output layer
		self.fc_out = nn.Linear(16, 1)
			

	def forward(self, x):
	
		x_eeg = x[:, 0:32, :]
		print(x_eeg.size())
		x_eog = x[:, 32:34, :]
		#x_emg = x[:, 34:36, :]
		
		#x_resp_ppg = x[:, 37:39, :]

		x_gsr = x[:, 36, :]
		x_temp = x[:, 39, :]

		x_gsr = x_gsr.contiguous()
		x_temp = x_temp.contiguous()

		x_gsr = x_gsr.view(x_gsr.size()[0], 1,  x_gsr.size()[1])
		x_temp = x_temp.view(x_temp.size()[0], 1,  x_temp.size()[1])

		x_others = torch.cat([x_temp, x_gsr], 1)

		#EEG
		x_eeg = self.features_eeg(x_eeg)
		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		x_eeg = self.features_eeg_flatten(x_eeg)

		#Skin temp and GSR
		x_others  = self.features_others(x_others)
		x_others = x_others.view(x_others.size(0), -1)
		x_others = self.features_others_flatten(x_others)
		
		concatenated_output = torch.cat([x_eeg, x_others], 1)

		#seq_out, _ = self.lstm(concatenated_output, (h0, c0))

		#output = self.fc_lstm(seq_out)
		output = F.relu(concatenated_output)
		output = F.sigmoid(self.fc_out(output))

		return output
	
		

		

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class model(nn.Module):
	
	def __init__(self):

		super(model, self).__init__()


		# Model 1
		self.conv1_1 = nn.Conv1d(40, 64, kernel_size = 16)		# 128 - 16 + 1 = 113
		self.conv1_2 = nn.Conv1d(64, 64, kernel_size = 16)		# 113 - 16 + 1 = 98
		self.conv2_1 = nn.Conv1d(64, 128, kernel_size = 16)		# 98 - 16 + 1 = 83
		self.conv2_2 = nn.Conv1d(128, 128, kernel_size = 16)	# 83 - 16 + 1 = 68
		self.conv3_1 = nn.Conv1d(128, 256, kernel_size = 16)	# 68 - 16 + 1 = 53
		self.conv3_2 = nn.Conv1d(256, 256, kernel_size = 16)	# 53 - 16 + 1 = 38
		self.conv4_1 = nn.Conv1d(256, 512, kernel_size = 16)	# 38 - 16 + 1 = 23
		self.conv4_2 = nn.Conv1d(512, 512, kernel_size = 16)	# 23 - 16 + 1 = 8
		self.fc1 = nn.Linear(8*512, 256)
		self.fc2 = nn.Linear(256, 4)

		# Model 2 

		# EEG network
		self.conv1_1_eeg = nn.Conv1d(32, 64, kernel_size = 16)		# 128 - 16 + 1 = 113
		self.conv1_2_eeg = nn.Conv1d(64, 64, kernel_size = 16)		# 113 - 16 + 1 = 98
		self.conv2_1_eeg = nn.Conv1d(64, 128, kernel_size = 16)		# 98 - 16 + 1 = 83
		self.conv2_2_eeg = nn.Conv1d(128, 128, kernel_size = 16)	# 83 - 16 + 1 = 68
		self.conv3_1_eeg = nn.Conv1d(128, 256, kernel_size = 16)	# 68 - 16 + 1 = 53
		self.fc1_eeg = nn.Linear(53*256, 128)
		self.fc2_eeg = nn.Linear(128, 64)							# Representation to be concatenated

		# EOG
		



	
	def forward(self, x):
		
		x = self.conv1_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv1_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = self.conv2_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv2_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = self.conv3_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv3_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = self.conv4_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv4_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training = self.training)
		x = F.sigmoid(self.fc2(x))
		
		return x

	def forward_multimodal(self, x):
	
		x_eeg = data[:, 0:32, :]
		x_eog = data[:, 32:34, :]
		x_emg = data[:, 34:36, :]
		x_gsr = data[:, 36, :]
		x_resp = data[:, 37, :]
		x_plet = data[:, 38, :]
		x_temp = data[:, 39, :]


		x = self.conv1_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv1_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = self.conv2_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv2_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = self.conv3_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv3_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = self.conv4_1(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)
		x = self.conv4_2(x)
		x = F.dropout(x, training = self.training)
		x = F.relu(x)

		x = x.view(x.size(0), -1)
		x = F.relu(self.fc1(x))
		x = F.dropout(x, training = self.training)
		x = F.sigmoid(self.fc2(x))
		
		return x	
		
		

		
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class model(nn.Module):
	
	def __init__(self):

		super(model, self).__init__()


				
		# EEG network
		self.conv1_1_eeg = nn.Conv1d(32, 48, kernel_size = 128)		# 384 - 128 + 1 = 257
		self.conv1_2_eeg = nn.Conv1d(48, 64, kernel_size = 128)		# 257 - 128 + 1 = 130
		self.conv2_1_eeg = nn.Conv1d(64, 64, kernel_size = 64)		# 130 - 64 + 1 = 67
		self.conv2_2_eeg = nn.Conv1d(64, 32, kernel_size = 16)		# 67 - 16 + 1 = 52
		self.conv3_1_eeg = nn.Conv1d(32, 16, kernel_size = 16)		# 52 - 16 + 1 = 37
		self.conv3_2_eeg = nn.Conv1d(16, 8, kernel_size = 16)		# 37 - 16 + 1 = 22
		self.conv4_1_eeg = nn.Conv1d(8, 4, kernel_size = 8)		# 22 - 8 + 1 = 15
		self.conv4_2_eeg = nn.Conv1d(4, 4, kernel_size = 8)		# 15 - 8 + 1 = 8
		self.fc1_eeg = nn.Linear(4*8, 16)
		self.fc2_eeg = nn.Linear(16, 8)				# Representation to be concatenated
		

		# Temp and GSR network
		self.conv1_1_temp_gsr = nn.Conv1d(2, 16, kernel_size = 192)	# 384 - 192 + 1 = 193
		self.conv2_1_temp_gsr = nn.Conv1d(16, 32, kernel_size = 128)	# 193 - 128 + 1 = 66
		self.conv3_1_temp_gsr = nn.Conv1d(32, 16, kernel_size = 64)	# 66 - 64 + 1 = 3
		self.fc1_temp_gsr = nn.Linear(16*3, 16)
		self.fc2_temp_gsr = nn.Linear(16, 8)				# Representation to be concatenated

		# FIX!!!!
		self.fc1_vi = nn.Linear(10*10*512, 512)
		self.fc2_vi = nn.Linear(512, 256)
		self.fc3_vi = nn.Linear(256, 128)

		self.fc1_au = nn.Linear(221*512, 512)
		self.fc2_au = nn.Linear(512, 256)
		self.fc3_au = nn.Linear(256, 128)

		self.lstm_vi = nn.LSTM(128, 32, 2, bidirectional=True, dropout=0.5)
		self.fc_vi = nn.Linear(32*2, 4)

		self.lstm_au = nn.LSTM(128, 32, 2, bidirectional=True, dropout=0.5)
		self.fc_au = nn.Linear(32*2, 4)
				

		# Output layer
		self.fc_out_1 = nn.Linear(16, 4)
		self.fc_out_arousal = nn.Linear(4, 1)


	
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

	def forward_multimodal_arousal(self, x):
	
		x_eeg = x[:, 0:32, :]
		x_eog = x[:, 32:34, :]
		#x_emg = x[:, 34:36, :]
		
		#x_resp_ppg = x[:, 37:39, :]

		x_gsr = x[:, 36, :]
		x_temp = x[:, 39, :]

		x_gsr = x_gsr.contiguous()
		x_temp = x_temp.contiguous()

		x_gsr = x_gsr.view(x_gsr.size()[0], 1,  x_gsr.size()[1])
		x_temp = x_temp.view(x_temp.size()[0], 1,  x_temp.size()[1])

				

		x_temp_gsr = torch.cat([x_temp, x_gsr], 1)


		#print(x_temp_gsr.size())


		#-------EEG
		x_eeg = self.conv1_1_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)
		x_eeg = self.conv1_2_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)

		x_eeg = self.conv2_1_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)
		x_eeg = self.conv2_2_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)

		x_eeg = self.conv3_1_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)
		x_eeg = self.conv3_2_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)

		x_eeg = self.conv4_1_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)
		x_eeg = self.conv4_2_eeg(x_eeg)
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = F.selu(x_eeg)


		
		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		#print(x_eeg.size())
		x_eeg = F.selu(self.fc1_eeg(x_eeg))
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = self.fc2_eeg(x_eeg)
		
		
		
		#------SKIN TEMP AND GSR
		x_temp_gsr = self.conv1_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.alpha_dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.selu(x_temp_gsr)

		x_temp_gsr = self.conv2_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.alpha_dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.selu(x_temp_gsr)

		x_temp_gsr = self.conv3_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.alpha_dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.selu(x_temp_gsr)
				
		x_temp_gsr = x_temp_gsr.view(x_temp_gsr.size(0), -1)
		x_temp_gsr = F.selu(self.fc1_temp_gsr(x_temp_gsr))
		x_temp_gsr = F.alpha_dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.selu(self.fc2_temp_gsr(x_temp_gsr))
		

		concatenated_output = torch.cat([x_eeg, x_temp_gsr], 1)

		# FIX!!!!
		seq_out_vi, _ = self.lstm_vi(seq_in_vi, (h0, c0))
		seq_out_au, _ = self.lstm_au(seq_in_au, (h0, c0))

		output = self.fc_out_1(concatenated_output)
		output = F.selu(output)
		output = F.alpha_dropout(output, training = self.training)
		output = F.sigmoid(self.fc_out_arousal(output))



		return output
	
		

		

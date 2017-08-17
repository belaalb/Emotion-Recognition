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
		self.conv3_2_eeg = nn.Conv1d(256, 64, kernel_size = 16)		# 53 - 16 + 1 = 38
		self.conv4_1_eeg = nn.Conv1d(64, 64, kernel_size = 16)		# 38 - 16 + 1 = 23
		self.conv4_2_eeg = nn.Conv1d(64, 16, kernel_size = 16)		# 23 - 16 + 1 = 8
		self.fc1_eeg = nn.Linear(16*8, 48)
		self.fc2_eeg = nn.Linear(48, 16)				# Representation to be concatenated

		# EOG network
		self.conv1_1_eog = nn.Conv1d(2, 16, kernel_size = 16)		# 128 - 16 + 1 = 113
		self.conv2_1_eog = nn.Conv1d(16, 64, kernel_size = 16)		# 113 - 16 + 1 = 98
		self.conv3_1_eog = nn.Conv1d(64, 128, kernel_size = 16)		# 98 - 16 + 1 = 83
		self.conv3_2_eog = nn.Conv1d(128, 128, kernel_size = 16)	# 83 - 16 + 1 = 68
		self.conv4_1_eog = nn.Conv1d(128, 256, kernel_size = 16)	# 68 - 16 + 1 = 53
		self.conv5_1_eog = nn.Conv1d(256, 64, kernel_size = 16)		# 53 - 16 + 1 = 38
		self.conv5_2_eog = nn.Conv1d(64, 64, kernel_size = 16)		# 38 - 16 + 1 = 23
		self.conv6_1_eog = nn.Conv1d(64, 16, kernel_size = 16)		# 23 - 16 + 1 = 8
		self.fc1_eog = nn.Linear(16*8, 48)
		self.fc2_eog = nn.Linear(48, 16)				# Representation to be concatenated

		# EMG network
		self.conv1_1_emg = nn.Conv1d(2, 16, kernel_size = 16)		# 128 - 16 + 1 = 113
		self.conv2_1_emg = nn.Conv1d(16, 64, kernel_size = 16)		# 113 - 16 + 1 = 98
		self.conv3_1_emg = nn.Conv1d(64, 128, kernel_size = 16)		# 98 - 16 + 1 = 83
		self.conv3_2_emg = nn.Conv1d(128, 128, kernel_size = 16)	# 83 - 16 + 1 = 68
		self.conv4_1_emg = nn.Conv1d(128, 256, kernel_size = 16)	# 68 - 16 + 1 = 53
		self.conv5_1_emg = nn.Conv1d(256, 64, kernel_size = 16)		# 53 - 16 + 1 = 38
		self.conv5_2_emg = nn.Conv1d(64, 64, kernel_size = 16)		# 38 - 16 + 1 = 23
		self.conv6_1_emg = nn.Conv1d(64, 16, kernel_size = 16)		# 23 - 16 + 1 = 8
		self.fc1_emg = nn.Linear(16*8, 48)
		self.fc2_emg = nn.Linear(48, 16)				# Representation to be concatenated
		
		# Resp and PPG network
		self.conv1_1_resp_ppg = nn.Conv1d(2, 16, kernel_size = 16)	# 128 - 16 + 1 = 113
		self.conv2_1_resp_ppg = nn.Conv1d(16, 64, kernel_size = 16)	# 113 - 16 + 1 = 98
		self.conv3_1_resp_ppg = nn.Conv1d(64, 128, kernel_size = 16)	# 98 - 16 + 1 = 83
		self.conv3_2_resp_ppg = nn.Conv1d(128, 128, kernel_size = 16)	# 83 - 16 + 1 = 68
		self.conv4_1_resp_ppg = nn.Conv1d(128, 256, kernel_size = 16)	# 68 - 16 + 1 = 53
		self.conv5_1_resp_ppg = nn.Conv1d(256, 64, kernel_size = 16)	# 53 - 16 + 1 = 38
		self.conv5_2_resp_ppg = nn.Conv1d(64, 64, kernel_size = 16)	# 38 - 16 + 1 = 23
		self.conv6_1_resp_ppg = nn.Conv1d(64, 16, kernel_size = 16)	# 23 - 16 + 1 = 8
		self.fc1_resp_ppg = nn.Linear(16*8, 48)
		self.fc2_resp_ppg = nn.Linear(48, 16)				# Representation to be concatenated
				

		# Temp and GSR network
		self.conv1_1_temp_gsr = nn.Conv1d(2, 16, kernel_size = 16)	# 128 - 16 + 1 = 113
		self.conv2_1_temp_gsr = nn.Conv1d(16, 64, kernel_size = 16)	# 113 - 16 + 1 = 98
		self.conv3_1_temp_gsr = nn.Conv1d(64, 128, kernel_size = 16)	# 98 - 16 + 1 = 83
		self.conv3_2_temp_gsr = nn.Conv1d(128, 128, kernel_size = 16)	# 83 - 16 + 1 = 68
		self.conv4_1_temp_gsr = nn.Conv1d(128, 256, kernel_size = 16)	# 68 - 16 + 1 = 53
		self.conv5_1_temp_gsr = nn.Conv1d(256, 64, kernel_size = 16)	# 53 - 16 + 1 = 38
		self.conv5_2_temp_gsr = nn.Conv1d(64, 64, kernel_size = 16)	# 38 - 16 + 1 = 23
		self.conv6_1_temp_gsr = nn.Conv1d(64, 16, kernel_size = 16)	# 23 - 16 + 1 = 8
		self.fc1_temp_gsr = nn.Linear(16*8, 48)
		self.fc2_temp_gsr = nn.Linear(48, 16)				# Representation to be concatenated



		# Output layer
		self.fc_out_1 = nn.Linear(16*5, 16)
		self.fc_out_arousal = nn.Linear(16, 3)


	
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
	
		x_eeg = x[:, 0:32, :]
		x_eog = x[:, 32:34, :]
		x_emg = x[:, 34:36, :]
		
		x_resp_ppg = x[:, 37:39, :]

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
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)
		x_eeg = self.conv1_2_eeg(x_eeg)
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)

		x_eeg = self.conv2_1_eeg(x_eeg)
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)
		x_eeg = self.conv2_2_eeg(x_eeg)
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)

		x_eeg = self.conv3_1_eeg(x_eeg)
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)
		x_eeg = self.conv3_2_eeg(x_eeg)
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)

		x_eeg = self.conv4_1_eeg(x_eeg)
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)
		x_eeg = self.conv4_2_eeg(x_eeg)
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = F.relu(x_eeg)

		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		x_eeg = F.relu(self.fc1_eeg(x_eeg))
		x_eeg = F.dropout(x_eeg, training = self.training)
		x_eeg = self.fc2_eeg(x_eeg)
		
		
		#-------EOG
		x_eog = self.conv1_1_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)

		x_eog = self.conv2_1_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)

		x_eog = self.conv3_1_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)
		x_eog = self.conv3_2_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)

		x_eog = self.conv4_1_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)

		x_eog = self.conv5_1_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)
		x_eog = self.conv5_2_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)

		x_eog = self.conv6_1_eog(x_eog)
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = F.relu(x_eog)

		x_eog = x_eog.view(x_eog.size(0), -1)
		x_eog = F.relu(self.fc1_eog(x_eog))
		x_eog = F.dropout(x_eog, training = self.training)
		x_eog = self.fc2_eog(x_eog)

		#-------EMG
		x_emg = self.conv1_1_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)

		x_emg = self.conv2_1_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)

		x_emg = self.conv3_1_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)
		x_emg = self.conv3_2_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)

		x_emg = self.conv4_1_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)

		x_emg = self.conv5_1_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)
		x_emg = self.conv5_2_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)

		x_emg = self.conv6_1_emg(x_emg)
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = F.relu(x_emg)

		x_emg = x_emg.view(x_emg.size(0), -1)
		x_emg = F.relu(self.fc1_emg(x_emg))
		x_emg = F.dropout(x_emg, training = self.training)
		x_emg = self.fc2_emg(x_emg)


		#------RESP AND PPG
		x_resp_ppg = self.conv1_1_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)

		x_resp_ppg = self.conv2_1_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)

		x_resp_ppg = self.conv3_1_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)
		x_resp_ppg = self.conv3_2_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)

		x_resp_ppg = self.conv4_1_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)

		x_resp_ppg = self.conv5_1_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)
		x_resp_ppg = self.conv5_2_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)

		x_resp_ppg = self.conv6_1_resp_ppg(x_resp_ppg)
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = F.relu(x_resp_ppg)

		x_resp_ppg = x_resp_ppg.view(x_resp_ppg.size(0), -1)
		x_resp_ppg = F.relu(self.fc1_resp_ppg(x_resp_ppg))
		x_resp_ppg = F.dropout(x_resp_ppg, training = self.training)
		x_resp_ppg = self.fc2_resp_ppg(x_resp_ppg)


		
		#------SKIN TEMP AND GSR
		x_temp_gsr = self.conv1_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)

		x_temp_gsr = self.conv2_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)

		x_temp_gsr = self.conv3_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)
		x_temp_gsr = self.conv3_2_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)

		x_temp_gsr = self.conv4_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)

		x_temp_gsr = self.conv5_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)
		x_temp_gsr = self.conv5_2_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)

		x_temp_gsr = self.conv6_1_temp_gsr(x_temp_gsr)
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = F.relu(x_temp_gsr)

		x_temp_gsr = x_temp_gsr.view(x_temp_gsr.size(0), -1)
		x_temp_gsr = F.relu(self.fc1_temp_gsr(x_temp_gsr))
		x_temp_gsr = F.dropout(x_temp_gsr, training = self.training)
		x_temp_gsr = self.fc2_temp_gsr(x_temp_gsr)
		

		concatenated_output = torch.cat([x_eeg, x_emg, x_eog, x_resp_ppg, x_temp_gsr], 1)
		
		#print(concatenated_output.size())

		output = self.fc_out_1(concatenated_output)
		output = F.relu(output)
		output = F.dropout(output, training = self.training)
		output = self.fc_out_arousal(output)
		output = F.Softmax(output)


		return output
	
		

		

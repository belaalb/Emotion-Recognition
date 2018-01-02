import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class model(nn.Module):
	
	def __init__(self):

		super(model, self).__init__()


				
		# EEG network
		self.conv1_1_eeg = nn.Conv2d(1, 8, (5, 128))			# 384 - 128 + 1 = 257; 32 - 5 + 1 = 28
		self.conv1_2_eeg = nn.Conv2d(8, 8, (5, 128))			# 257 - 128 + 1 = 130; 28 - 5 + 1 = 24
		self.conv2_1_eeg = nn.Conv2d(8, 16, (5, 64))			# 130 - 64 + 1 = 67; 24 - 5 + 1 = 20
		self.conv2_2_eeg = nn.Conv2d(16, 16, (5, 32))			# 67 - 32 + 1 = 36; 20 - 5 + 1 = 16
		self.conv3_1_eeg = nn.Conv2d(16, 32, (5, 16))			# 36 - 16 + 1 = 21; 16 - 5 + 1 = 12 
		self.conv3_2_eeg = nn.Conv2d(32, 16, (5, 8))			# 21 - 8 + 1 = 14; 12 - 5 + 1 = 8
		self.conv4_1_eeg = nn.Conv2d(16, 8, (5, 8))			# 14 - 8 + 1 = 7; 8 - 5 + 1 = 4 
		self.fc1_eeg = nn.Linear(8*7*4, 60)
		self.fc2_eeg = nn.Linear(60, 40)					# Representation to be concatenated

		# Output layer
		self.fc_out_1_eeg = nn.Linear(40, 20)
		self.fc_out_2_eeg = nn.Linear(20, 10)
		self.fc_out_eeg = nn.Linear(10, 2)
		

		# Temp and GSR network
		self.conv1_1_temp_gsr = nn.Conv1d(2, 16, kernel_size = 192)	# 384 - 192 + 1 = 193
		self.conv2_1_temp_gsr = nn.Conv1d(16, 32, kernel_size = 128)	# 193 - 128 + 1 = 66
		self.conv3_1_temp_gsr = nn.Conv1d(32, 16, kernel_size = 64)	# 66 - 64 + 1 = 3
		self.fc1_temp_gsr = nn.Linear(16*3, 32)
		self.fc2_temp_gsr = nn.Linear(32, 20)				# Representation to be concatenated
				

		# Output layer
		self.fc_out_1_temp_gsr = nn.Linear(20, 10)
		self.fc_out_2_temp_gsr = nn.Linear(10, 5)
		self.fc_out_temp_gsr = nn.Linear(5, 2)


	def forward_multimodal(self, x):
	
		x_eeg = x[:, 0:32, :]
		x_eeg = x_eeg.contiguous()
		x_eeg = x_eeg.view(x_eeg.size()[0], 1, x_eeg.size()[1], x_eeg.size()[2])
		
		#x_eog = x[:, 32:34, :]
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


		x_eeg = x_eeg.view(x_eeg.size(0), -1)
		x_eeg = F.selu(self.fc1_eeg(x_eeg))
		x_eeg = F.alpha_dropout(x_eeg, training = self.training)
		x_eeg = self.fc2_eeg(x_eeg)

		output_eeg = self.fc_out_1_eeg(x_eeg)
		output_eeg = F.alpha_dropout(output_eeg, training = self.training)
		output_eeg = F.selu(output_eeg)
		output_eeg = self.fc_out_2_eeg(output_eeg)
		output_eeg = F.alpha_dropout(output_eeg, training = self.training)
		output_eeg = F.selu(output_eeg)
		output_eeg = F.softmax(self.fc_out_eeg(output_eeg))
		
		
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
		

		output_temp_gsr = self.fc_out_1_temp_gsr(x_temp_gsr)
		output_temp_gsr = F.alpha_dropout(output_temp_gsr, training = self.training)
		output_temp_gsr = F.selu(output_temp_gsr)
		output_temp_gsr = self.fc_out_2_temp_gsr(output_temp_gsr)
		output_temp_gsr = F.alpha_dropout(output_temp_gsr, training = self.training)
		output_temp_gsr = F.selu(output_temp_gsr)
		output_temp_gsr = F.softmax(self.fc_out_temp_gsr(output_temp_gsr))



		return output_eeg, output_temp_gsr
	
		

		

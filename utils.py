import numpy as np
import random
import pickle
import os, sys
import h5py


def load_dataset_per_subject(sub = 1, main_dir = 'data_preprocessed_python/'):

	if (sub < 10):
		sub_code = str('s0' + str(sub) + '.dat')
	else:
		sub_code = str('s' + str(sub) + '.dat')	

	subject_path = os.path.join(main_dir, sub_code)
	subject = pickle.load(open(subject_path, 'rb'), encoding = 'latin1')
	labels = subject['labels']
	data = subject['data'][:, :, 3*128-1:-1] # Excluding the first 3s of baseline
		
	return data, labels


def split_data_per_subject(sub = 1, segment_duration = 1, sampling_rate = 128, main_dir = '/home/isabela/emot_recog_class/data_preprocessed_python/'):

	'''
	TO DO:
	- Increase window length
	- Overlapping windows
 
	'''

	data, labels = load_dataset_per_subject(sub, main_dir)
	number_of_samples = segment_duration*sampling_rate
	number_of_trials = data.shape[0]
	number_of_segments = data.shape[2]/number_of_samples

	number_of_final_examples = number_of_trials*number_of_segments

	final_data = []
	final_labels = []

	for trial in range(0, number_of_trials):

		data_to_split = data[trial, :, :]
		data_to_split = np.reshape(data_to_split, (number_of_segments, data.shape[1], number_of_samples))
		final_data.append(data_to_split)

		label_to_repeat = labels[trial, :]
		label_repeated = np.tile(label_to_repeat, (number_of_segments, 1))
		final_labels.append(label_repeated)

	final_data_out = np.reshape(np.asarray(final_data), (number_of_final_examples, data.shape[1], number_of_samples))
	final_labels_out = np.reshape(np.asarray(final_labels), (number_of_final_examples, labels.shape[1]))


	return final_data_out, final_labels_out


def create_hdf(subjects_number = 32, hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	complete_data = []
	complete_labels = []

	for sub in range(1, subjects_number+1):

		data_sub, labels_sub = split_data_per_subject(sub)
		complete_data.append(data_sub)
		complete_labels.append(labels_sub)

	complete_data = np.asarray(complete_data)
	complete_labels = np.asarray(complete_labels)

	print(complete_data.shape)
	print(complete_labels.shape)	

	complete_dataset_file = h5py.File(hdf_filename, 'w')
	complete_dataset = complete_dataset_file.create_dataset('data', data = complete_data)
	complete_dataset = complete_dataset_file.create_dataset('labels', data = complete_labels)
	complete_dataset_file.close()




def merge_shuffle_norm_split_tvt_store_as_hdf(hdf_filename_to_read = 'DEAP_dataset_subjects_list.hdf', hdf_filename_to_save_train = 'DEAP_dataset_train.hdf', hdf_filename_to_save_valid = 'DEAP_dataset_valid.hdf', hdf_filename_to_save_test = 'DEAP_dataset_test.hdf'):

	#tvt: train, valid, test

	data, labels = read_hdf(hdf_filename_to_read)
	print(data.shape)
	print(labels.shape)	

	number_of_examples = data.shape[0]*data.shape[1]
	print(number_of_examples)
	
	data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
	labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))

	# Shuffling

	idxs = np.random.permutation(np.arange(number_of_examples))
	data = data[idxs, :, :]
	labels = labels[idxs, :]


	labels_arousal_val = np.zeros([number_of_examples, 2]);
	#labels_val = np.zeros([number_of_examples, 3]);


	labels_arousal_val[(1 <= labels[:, 0]) & (labels[:, 0] <= 3.5), 0] = 0
	labels_arousal_val[(3.5 < labels[:, 0]) & (labels[:, 0] <= 6.5), 0] = 1 
	labels_arousal_val[(6.5 < labels[:, 0]) & (labels[:, 0] <= 9), 0] = 2
	

	labels_arousal_val[(1 <= labels[:, 1]) & (labels[:, 1] <= 3.5), 1] = 0
	labels_arousal_val[(3.5 < labels[:, 1]) & (labels[:, 1] <= 6.5), 1] = 1 
	labels_arousal_val[(6.5 < labels[:, 1]) & (labels[:, 1] <= 9), 1] = 2


	data_train = data[0:int(0.7*number_of_examples), :, :]
	data_valid = data[int(0.7*number_of_examples):int(0.9*number_of_examples), :, :]
	data_test = data[int(0.9*number_of_examples):-1, :, :]
	
	labels_arousal_val_train = labels_arousal_val[0:int(0.7*number_of_examples), :]
	labels_arousal_val_valid = labels_arousal_val[int(0.7*number_of_examples):int(0.9*number_of_examples), :]
	labels_arousal_val_test = labels_arousal_val[int(0.9*number_of_examples):-1, :]
	
	dataset_file_train = h5py.File(hdf_filename_to_save_train, 'w')
	dataset_train = dataset_file_train.create_dataset('data', data = data_train)
	dataset_train = dataset_file_train.create_dataset('labels_arousal_val', data = labels_arousal_val_train)
	dataset_file_train.close()

	dataset_file_valid = h5py.File(hdf_filename_to_save_valid, 'w')
	dataset_valid = dataset_file_valid.create_dataset('data', data = data_valid)
	dataset_valid = dataset_file_valid.create_dataset('labels_arousal_val', data = labels_arousal_val_valid)
	dataset_file_valid.close()

	dataset_file_test = h5py.File(hdf_filename_to_save_test, 'w')
	dataset_test = dataset_file_test.create_dataset('data', data = data_test)
	dataset_test = dataset_file_test.create_dataset('labels_arousal_val', data = labels_arousal_val_test)
	dataset_file_test.close()


def read_hdf(hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	labels = open_file['labels']


	return data, labels

def read_hdf_processed_labels(hdf_filename = 'DEAP_dataset_train.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	labels_arousal_val = open_file['labels_arousal_val']

	#open_file.close()

	return data, labels_arousal_val

def read_hdf_processed_labels_return_size(hdf_filename = 'DEAP_dataset_train.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	length = data.shape[0]	

	open_file.close()

	return length

def read_hdf_processed_labels_idx(idx, hdf_filename = 'DEAP_dataset_train.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data'][idx]
	labels_arousal_val = open_file['labels_arousal_val'][idx]

	open_file.close()

	return data, labels_arousal_val

def calculate_weights(root = "/home/isabela/Desktop/emot_recog_class/DEAP_dataset", step = "train"):

	if (step == "train"):
		dataset_filename = root + "_train.hdf"
	elif (step == "valid"):
		dataset_filename = root + "_valid.hdf"
	else:
		dataset_filename = root + "_test.hdf"

	_, labels_arousal_val = read_hdf_processed_labels(dataset_filename)

	p0 = sum(labels_arousal_val[:, 0] == 0) / labels_arousal_val.shape[0] 	
	p1 = sum(labels_arousal_val[:, 0] == 1) / labels_arousal_val.shape[0]
	p2 = sum(labels_arousal_val[:, 0] == 2) / labels_arousal_val.shape[0]

	probs = [p0, p1, p2]

	reciprocal_weights = [0] * len(labels_arousal_val) 

	for idx in range(len(labels_arousal_val)):
		reciprocal_weights[idx] = probs[int(labels_arousal_val[idx, 0])]

	length = len(labels_arousal_val)

	return reciprocal_weights, length



if __name__ == '__main__':

	#merge_shuffle_norm_split_tvt_store_as_hdf()
	
	data, labels_arousal_val = read_hdf_processed_labels('/home/isabela/Desktop/emot_recog_class/DEAP_dataset_train.hdf')
	#print(data.shape)
	#print(labels_arousal.shape)
	#print(labels_val.shape)


	print('Label 0', sum(labels_arousal_val[:, 0] == 0))	
	print('Label 1', sum(labels_arousal_val[:, 0] == 1))
	print('Label 2', sum(labels_arousal_val[:, 0] == 2))
	#print(labels_arousal_val[234:255, 1])

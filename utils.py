import numpy as np
import random
import pickle
import os, sys
import h5py

from numpy.lib.stride_tricks import as_strided


def load_dataset_per_subject(sub = 1, main_dir = '/home/isabela/Desktop/emot_recog_class/data_preprocessed_python/'):

	if (sub < 10):
		sub_code = str('s0' + str(sub) + '.dat')
	else:
		sub_code = str('s' + str(sub) + '.dat')	

	subject_path = os.path.join(main_dir, sub_code)
	subject = pickle.load(open(subject_path, 'rb'), encoding = 'latin1')
	labels = subject['labels']
	data = subject['data'][:, :, 3*128-1:-1] # Excluding the first 3s of baseline
		
	return data, labels

def strided_app(a, window_length, S):

	nrows = ((len(a) - window_length) // S ) + 1
	n = a.strides[0]
	
	return as_strided(a, shape = (nrows, window_length), strides = (S * n, n))


def add_gaussian_noise(data, labels, n):
	
	'''
	Data augmentation:
	Select n examples from data to add gaussian noise  
	Returns data merged with the new examples and labels merged with the respective new labels
	'''

	idx = np.where(labels == 0)

	idx = np.random.permutation(idx)
	
	data_augmented = data[idx, :]
	labels_augmented = labels[idx, :]

	gaussian_noise = np.random.normal(0, 0.01, data_augmented.shape)
	data_augmented = data_augmented + gaussian_noise

	new_data = np.concatenate(data, data_augmented)
	new_labels = np.concatenate(labels, labels_augmented)

	return new_data, new_labels
	

def labels_quantization(labels):

	labels_val = np.zeros([labels.shape[0], 1]);

	labels_val[(1 <= labels[:, 0]) & (labels[:, 0] <= 5), 0] = 0
	labels_val[(5 < labels[:, 0]) & (labels[:, 0] <= 9), 0] = 1 

	return labels_val



def split_data_per_subject(sub = 1, segment_duration = 3, sampling_rate = 128, main_dir = '/home/isabela/Desktop/emot_recog_class/data_preprocessed_python/'):

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


def split_data_per_subject_overlapping(sub = 1, window_duration = 3, sampling_rate = 128, strides = 1, main_dir = '/home/isabela/Desktop/emot_recog_class/data_preprocessed_python/'):


	data, labels = load_dataset_per_subject(sub, main_dir)
	window_samples = window_duration * sampling_rate
	strides_samples = strides * sampling_rate
	number_of_trials = data.shape[0]
	number_of_channels = data.shape[1]
 
	final_data = []
	final_labels = []

	for trial in range(0, number_of_trials):

		examples_all_channels = []
	
		for channel in range(0, number_of_channels):

			data_to_split = data[trial, channel, :]
			data_chunks = strided_app(data_to_split, window_samples, strides_samples)
			examples_per_channel = np.reshape(data_chunks, (data_chunks.shape[0], 1, window_samples))
			examples_all_channels.append(examples_per_channel)
		
		data_examples_in_row = np.reshape(np.asarray(examples_all_channels), (data_chunks.shape[0], data.shape[1], window_samples))
		final_data.append(data_examples_in_row)

		label_to_repeat = labels[trial, :]
		label_repeated = np.tile(label_to_repeat, (data_chunks.shape[0], 1))
		final_labels.append(label_repeated)

	number_of_final_examples = number_of_trials * data_chunks.shape[0]

	final_data_out = np.reshape(np.asarray(final_data), (number_of_final_examples, data.shape[1], window_samples))
	final_labels_out = np.reshape(np.asarray(final_labels), (number_of_final_examples, labels.shape[1]))

	return final_data_out, final_labels_out


def create_hdf(subjects_number = 32, hdf_filename = '/home/isabela/Desktop/emot_recog_class/less_signals_3s_overlap/DEAP_dataset_subjects_list.hdf'):

	complete_data = []
	complete_labels = []

	for sub in range(1, subjects_number+1):

		data_sub, labels_sub = split_data_per_subject_overlapping(sub)
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


def merge_shuffle_norm_split_tvt_store_as_hdf(hdf_filename_to_read = '/home/isabela/Desktop/emot_recog_class/less_signals_3s_overlap/DEAP_dataset_subjects_list.hdf', hdf_filename_to_save_train = '/home/isabela/Desktop/emot_recog_class/less_signals_3s_overlap/DEAP_dataset_train.hdf', hdf_filename_to_save_valid = '/home/isabela/Desktop/emot_recog_class/less_signals_3s_overlap/DEAP_dataset_valid.hdf'):

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

	# Rescaling to [0; 1]

	max_per_channel = np.amax(np.amax(data, axis = 2), axis = 0)
	print(max_per_channel)
	min_per_channel = np.amin(np.amin(data, axis = 2), axis = 0)
	print(min_per_channel)

	data = (data - min_per_channel) / (max_per_channel - min_per_channel)

	labels_val = labels_quantization(labels)

	data_train = data[0:int(0.9*number_of_examples), :, :]
	data_valid = data[int(0.9*number_of_examples):-1, :, :]

	
	labels_val_train = labels_val[0:int(0.9*number_of_examples), :]
	labels_val_valid = labels_val[int(0.9*number_of_examples):-1, :]

	
	dataset_file_train = h5py.File(hdf_filename_to_save_train, 'w')
	dataset_train = dataset_file_train.create_dataset('data', data = data_train)
	dataset_train = dataset_file_train.create_dataset('labels_val', data = labels_val_train)
	dataset_file_train.close()

	dataset_file_valid = h5py.File(hdf_filename_to_save_valid, 'w')
	dataset_valid = dataset_file_valid.create_dataset('data', data = data_valid)
	dataset_valid = dataset_file_valid.create_dataset('labels_val', data = labels_val_valid)
	dataset_file_valid.close()



def read_hdf(hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	labels = open_file['labels']


	return data, labels

def read_hdf_processed_labels(hdf_filename = 'DEAP_dataset_train.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	labels_val = open_file['labels_val']

	#open_file.close()

	return data, labels_val


def read_hdf_processed_labels_return_size(hdf_filename = 'DEAP_dataset_train.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	length = data.shape[0]	

	open_file.close()

	return length


def read_hdf_processed_labels_idx(idx, noisy = True, hdf_filename = 'DEAP_dataset_train.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data'][idx]
	labels_val = open_file['labels_val'][idx]

	if noisy:
		if labels_val == 0:
			data = data + np.random.gaussian(0, 0.01, data.shape)

	open_file.close()

	return data, labels_val


def calculate_weights(root = "/home/isabela/Desktop/emot_recog_class/less_signals_3s_overlap/DEAP_dataset", step = "train"):

	if (step == "train"):
		dataset_filename = root + "_train.hdf"
	elif (step == "valid"):
		dataset_filename = root + "_valid.hdf"
	else:
		dataset_filename = root + "_test.hdf"

	_, labels_val = read_hdf_processed_labels(dataset_filename)

	p0 = sum(labels_val[:, 1] == 0) / labels_val.shape[0] 	
	p1 = sum(labels_val[:, 1] == 1) / labels_val.shape[0]

	probs = [p0, p1]

	reciprocal_weights = [0] * len(labels_val) 

	for idx in range(len(labels_val)):
		reciprocal_weights[idx] = probs[int(labels_val[idx, 1])]

	length = len(labels_val)

	return reciprocal_weights, length


if __name__ == '__main__':

	#create_hdf()

	merge_shuffle_norm_split_tvt_store_as_hdf()
	
	data, labels_val = read_hdf_processed_labels('/home/isabela/Desktop/emot_recog_class/less_signals_3s_overlap/DEAP_dataset_train.hdf')
	print(data.shape)
	print(labels_val.shape)


	print('Label 0', sum(labels_val[:, 0] == 0))	
	print('Label 1', sum(labels_val[:, 0] == 1))
	print(labels_val[234:255, 1])

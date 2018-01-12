import numpy as np
import random
import pickle
import os, sys
import h5py


def load_dataset_per_subject(sub = 1, main_dir = './data_preprocessed_python/'):

	if (sub < 10):
		sub_code = str('s0' + str(sub) + '.dat')
	else:
		sub_code = str('s' + str(sub) + '.dat')	

	subject_path = os.path.join(main_dir, sub_code)
	subject = pickle.load(open(subject_path, 'rb'), encoding = 'latin1')
	labels = subject['labels']
	data = subject['data'][:, :, 3*128-1:-1] # Excluding the first 3s of baseline

	return data, labels


def split_data_per_subject(sub = 1, segment_duration = 1, seq_length = 10, sampling_rate = 128, main_dir = './data_preprocessed_python/'):

	data, labels = load_dataset_per_subject(sub, main_dir)
	n_points = segment_duration*sampling_rate
	n_trials = data.shape[0]
	n_segments = data.shape[2]//n_points

	n_examples = n_trials * n_segments

	data_non_seq = []
	labels_expanded = []

	for trial in range(n_trials):

		data_to_split = data[trial, :, :]
		data_to_split = np.reshape(data_to_split, (n_segments, data.shape[1], n_points))
		data_non_seq.append(data_to_split)

		label_to_repeat = labels[trial, :]
		label_repeated = np.tile(label_to_repeat, (n_segments, 1))
		labels_expanded.append(label_repeated)

	data_non_seq = np.reshape(np.asarray(data_non_seq), (n_examples, data.shape[1], n_points))
	labels = np.reshape(np.asarray(labels_expanded), (n_examples, labels.shape[1]))

	data_non_seq_res = rescale(data_non_seq)	

	data_seq = []

	for ex in range(n_examples - seq_length):
		
		example_seq = []

		for idx in range(seq_length):
			
			example_seq.append(data_non_seq_res[ex + idx, :, :])

		example_seq = np.reshape(np.asarray(example_seq), (seq_length, data.shape[1], n_points))
		data_seq.append(example_seq)
	
	data_seq = np.reshape(np.asarray(data_seq), (len(data_seq), seq_length, data.shape[1], n_points))

	labels = labels_quantization(labels)

	return data_seq, labels

def rescale(data_non_seq):

	max_per_channel = np.max(np.max(data_non_seq, axis = 2), axis = 0)
	min_per_channel = np.min(np.min(data_non_seq, axis = 2), axis = 0)

	for sample in data_non_seq:
		for channel in range(0, data_non_seq.shape[1]):
			sample[channel, :] = (sample[channel, :] - min_per_channel[channel]) / (max_per_channel[channel] - min_per_channel[channel])
	
	return data_non_seq

def labels_quantization(labels):

	median_val = np.median(labels[:, 0])
	median_arousal = np.median(labels[:, 1])

	labels_val = np.zeros(labels.shape[0])
	labels_arousal = np.zeros(labels.shape[0])

	labels_val[(1 <= labels[:, 0]) & (labels[:, 0] <= median_val)] = 0
	labels_val[(median_val < labels[:, 0]) & (labels[:, 0] <= 9)] = 1

	labels_arousal[(1 <= labels[:, 1]) & (labels[:, 1] <= median_arousal)] = 0
	labels_arousal[(median_arousal < labels[:, 1]) & (labels[:, 1] <= 9)] = 1

	labels[:, 0] = labels_val
	labels[:, 1] = labels_arousal
	labels = labels[:, 0:2]

	return labels	
	
	
def merge_all_subjects(n_subjects = 32, n_sub_valid = 3, root_filename = 'DEAP_complete_sequence_'):

	filename_train = root_filename + 'train.hdf'
	filename_valid = root_filename + 'valid.hdf'

	complete_dataset_train = h5py.File(filename_train, 'w')

	for sub in range(1, n_subjects - n_sub_valid + 1):

		data_sub, labels_sub = split_data_per_subject(sub, seq_length = 10)
		data_key = str('data_s' + str(sub))
		labels_key = str('labels_s' + str(sub))
		complete_dataset_train[data_key] = data_sub
		complete_dataset_train[labels_key] = labels_sub
	
	complete_dataset_train.close()


	complete_dataset_valid = h5py.File(filename_valid, 'w')

	for sub in range(n_subjects - n_sub_valid + 1, n_subjects + 1):

		data_sub, labels_sub = split_data_per_subject(sub, seq_length = 10)
		data_key = str('data_s' + str(sub))
		labels_key = str('labels_s' + str(sub))
		complete_dataset_valid[data_key] = data_sub
		complete_dataset_valid[labels_key] = labels_sub
	
	complete_dataset_valid.close()

if __name__ == '__main__':
	
	merge_all_subjects()



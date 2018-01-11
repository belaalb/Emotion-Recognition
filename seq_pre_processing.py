import numpy as np
import random
import pickle
import os, sys
import h5py

from numpy.lib.stride_tricks import as_strided


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


def split_data_per_subject(sub = 1, segment_duration = 1, seq_length = 20, sampling_rate = 128, main_dir = './data_preprocessed_python/'):

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


	data_seq = []

	for ex in range(n_examples - seq_length):
		example_seq = []
		for idx in range(seq_length):
			example_seq.append(data_non_seq[ex + idx, :, :])
			
		data_seq.append(example_seq)

	print(len(data_seq[0]))

	data_seq = np.reshape(np.asarray(data_seq, (len(data_seq), seq_length, data.shape[1], n_points)))

	return data_seq, labels

def merge_all_subjects(subjects_number = 32, hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	complete_data = []
	complete_labels = []

	for sub in range(1, subjects_number+1):

		data_sub, labels_sub = split_data_per_subject_overlapping(sub)
		complete_data.append(data_sub)
		complete_labels.append(labels_sub)

	complete_data = np.asarray(complete_data)
	complete_labels = np.asarray(complete_labels)

	complete_dataset_file = h5py.File(hdf_filename, 'w')
	complete_dataset = complete_dataset_file.create_dataset('data', data = complete_data)
	complete_dataset = complete_dataset_file.create_dataset('labels', data = complete_labels)
	complete_dataset_file.close()

def labels_quantization(labels):

	labels_val = np.zeros([labels.shape[0], 1]);

	median = np.median(labels)

	print(median)

	labels_val[(1 <= labels[:, 0]) & (labels[:, 0] <= median), 0] = 0
	labels_val[(median < labels[:, 0]) & (labels[:, 0] <= 9), 0] = 1 

	return labels_val

#def rescale():

if __name__ == '__main__':
	split_data_per_subject()


	


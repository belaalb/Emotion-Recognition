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

		example_seq = np.reshape(np.asarray(example_seq), (seq_length, data.shape[1], n_points))
		data_seq.append(example_seq)

	data_seq = np.reshape(np.asarray(data_seq), (len(data_seq), seq_length, data.shape[1], n_points))

	return data_seq, labels

def merge_all_subjects(n_subjects = 32, hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	complete_dataset_file = h5py.File(hdf_filename, 'w')

	for sub in range(1, n_subjects + 1):

		data_sub, labels_sub = split_data_per_subject(sub, seq_length = 20)
		data_key = str('data_s' + str(sub))
		labels_key = str('labels_s' + str(sub))
		complete_dataset = complete_dataset_file.create_dataset(data_key, data = data_sub)
		complete_dataset = complete_dataset_file.create_dataset(labels_key, data = labels_sub)
	
	complete_dataset_file.close()

def rescale_labels_quantization(n_subjects = 32, seq_length = 20, hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	data = h5py.File(hdf_filename, 'r')

	all_data = []
	all_labels = []

	for sub in range(1, n_subjects + 1):
		data_key = str('data_s' + str(sub))		
		labels_key = str('labels_s' + str(sub))
		all_data.append(data[data_key])		
		all_labels.append(data[labels_key][:, :-1*seq_length])
	
	all_data = np.asarray(all_data)
	print(all_data.shape)
	all_labels = np.asarray(all_labels)
	print(all_labels.shape)	

	# Labels quantization by the median
	valence_median = median(all_labels[:, 0])
	arousal_median = median(all_labels[:, 1])

	return labels_val

#def rescale():

if __name__ == '__main__':
	merge_all_subjects()
	rescale_labels_quantization()

	


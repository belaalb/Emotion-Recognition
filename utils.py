import numpy as np
import cPickle
import os, sys


def load_dataset_per_subject(sub = 1, main_dir = 'data_preprocessed_python/'):

	if (sub < 10):
		sub_code = str('s0' + str(sub) + '.dat')
	else:
		sub_code = str('s' + str(sub) + '.dat')	

	subject_path = os.path.join(main_dir, sub_code)
	subject = cPickle.load(open(subject_path, 'rb'))
	labels = subject['labels']
	data = subject['data']
		
	return data, labels

def split_data_per_subject(segment_duration = 1, sampling_rate = 128, sub = 1, main_dir = 'data_preprocessed_python/'):
	
	data, labels = load_dataset_per_subject(sub, main_dir)
	number_of_samples = segment_duration*sampling_rate
	number_of_trials = data.shape[0]
	number_of_segments = data.shape[2]/number_of_samples

	number_of_final_examples = number_of_trials*number_of_segments

	final_data = []
	final_labels = []

	for trial in xrange(0, number_of_trials):

		data_to_split = data[trial, :, :]
		data_to_split = np.reshape(data_to_split, (number_of_segments, data.shape[1], number_of_samples))
		final_data.append(data_to_split)

		label_to_repeat = labels[trial, :]
		label_repeated = np.tile(label_to_repeat, (number_of_segments, 1))
		final_labels.append(label_repeated)

	final_data_out = np.reshape(np.asarray(final_data), (number_of_final_examples, data.shape[1], number_of_samples))
	final_labels_out = np.reshape(np.asarray(final_labels), (number_of_final_examples, labels.shape[1]))	

	return final_data_out, final_labels_out	

if __name__ == '__main__':

	final_data, final_labels = split_data_per_subject()
	print(final_data.shape)
	#print(final_data[39].shape)
	print('labels')
	print(final_labels.shape)
	#print(final_labels[39].shape)
	
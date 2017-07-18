import numpy as np
import cPickle
import os, sys
import h5py


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

def split_data_per_subject(sub = 1, segment_duration = 1, sampling_rate = 128, main_dir = 'data_preprocessed_python/'):
	
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


def create_hdf(subjects_number = 32, hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	complete_data = []
	complete_labels = []

	for sub in xrange(1, subjects_number+1):

		data_sub, labels_sub = split_data_per_subject(sub)

			
		complete_data.append(data_sub)
		complete_labels.append(labels_sub)

	complete_data = np.asarray(complete_data)
	complete_labels = np.asarray(complete_labels)	

	complete_dataset_file = h5py.File(hdf_filename, 'w')
	complete_dataset = complete_dataset_file.create_dataset('data', data = complete_data)
	complete_dataset = complete_dataset_file.create_dataset('labels', data = complete_labels)
	complete_dataset_file.close()



def read_hdf(hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	labels = open_file['labels']

	return data, labels

def merge_all_subjects_store_as_hdf(hdf_filename_to_read = 'DEAP_dataset_subjects_list.hdf', hdf_filename_to_save = 'DEAP_dataset.hdf'):

	data, labels = read_hdf(hdf_filename_to_read)
	
	data_new = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
	labels_new = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))

	complete_dataset_file = h5py.File(hdf_filename_to_save, 'w')
	complete_dataset = complete_dataset_file.create_dataset('data', data = data_new)
	complete_dataset = complete_dataset_file.create_dataset('labels', data = labels_new)
	complete_dataset_file.close()



if __name__ == '__main__':

	#final_data, final_labels = split_data_per_subject()
	#print(final_data.shape)
	#print(final_data[39].shape)
	#print('labels')
	#print(final_labels.shape)
	#print(final_labels[39].shape)
	
	create_hdf(2)

	merge_all_subjects_store_as_hdf()

	data, labels = read_hdf('DEAP_dataset_subjects_list.hdf')
	print(labels[62])
import numpy as np
import random
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


def merge_all_subjects_store_as_hdf(hdf_filename_to_read = 'DEAP_dataset_subjects_list.hdf', hdf_filename_to_save = 'DEAP_dataset_aux.hdf'):

	data, labels = read_hdf(hdf_filename_to_read)
	
	data_new = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
	labels_new = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))

	complete_dataset_file = h5py.File(hdf_filename_to_save, 'w')
	complete_dataset = complete_dataset_file.create_dataset('data', data = data_new)
	complete_dataset = complete_dataset_file.create_dataset('labels', data = labels_new)
	complete_dataset_file.close()


def merge_all_subjects_shuffle_norm_split_store_as_hdf(hdf_filename_to_read = 'DEAP_dataset_subjects_list.hdf', hdf_filename_to_save = 'DEAP_dataset_aux.hdf'):

	data, labels = read_hdf(hdf_filename_to_read)
	
	data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
	labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))

	# Shuffling
	
	number_ef_examples = data.shape[0]
	random_seq = range(0, number_ef_examples) 
	random.shuffle(random_seq)
	data = data[random_seq, :, :]
	labels = labels[random_seq, :]


	# Normalizing labels between 0 and 1

	max_val_0 = np.max(labels[:, 0])
	min_val_0 = np.min(labels[:, 0])
	labels[:, 0] = (labels[:, 0] - min_val_0)/(max_val_0 - min_val_0)

	max_val_1 = np.max(labels[:, 1])
	min_val_1 = np.min(labels[:, 1])
	labels[:, 1] = (labels[:, 1] - min_val_1)/(max_val_1 - min_val_1)

	max_val_2 = np.max(labels[:, 2])
	min_val_2 = np.min(labels[:, 2])
	labels[:, 2] = (labels[:, 2] - min_val_2)/(max_val_2 - min_val_2)
	
	max_val_3 = np.max(labels[:, 3])
	min_val_3 = np.min(labels[:, 3])
	labels[:, 3] = (labels[:, 3] - min_val_3)/(max_val_3 - min_val_3)

	
	# Spliting the dataset according to different signal modalities

	data_eeg = data[:, 0:32, :]
	data_eog = data[:, 32:34, :]
	data_emg = data[:, 34:36, :]
	data_gsr = data[:, 36, :]
	data_resp = data[:, 37, :]
	data_plet = data[:, 38, :]
	data_temp = data[:, 39, :]

	complete_dataset_file = h5py.File(hdf_filename_to_save, 'w')
	complete_dataset = complete_dataset_file.create_dataset('eeg', data = data_eeg)
	complete_dataset = complete_dataset_file.create_dataset('eog', data = data_eog)
	complete_dataset = complete_dataset_file.create_dataset('emg', data = data_emg)
	complete_dataset = complete_dataset_file.create_dataset('gsr', data = data_gsr)
	complete_dataset = complete_dataset_file.create_dataset('resp', data = data_resp)
	complete_dataset = complete_dataset_file.create_dataset('plet', data = data_plet)
	complete_dataset = complete_dataset_file.create_dataset('temp', data = data_temp)
	complete_dataset = complete_dataset_file.create_dataset('labels', data = labels)
	complete_dataset_file.close()


def merge_shuffle_norm_split_tvt_store_as_hdf(hdf_filename_to_read = 'DEAP_dataset_subjects_list.hdf', hdf_filename_to_save_train = 'DEAP_dataset_train.hdf', hdf_filename_to_save_valid = 'DEAP_dataset_valid.hdf', hdf_filename_to_save_test = 'DEAP_dataset_test.hdf'):

	#tvt: train, valid, test

	data, labels = read_hdf(hdf_filename_to_read)
	
	data = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2], data.shape[3]))
	labels = np.reshape(labels, (labels.shape[0]*labels.shape[1], labels.shape[2]))

	# Shuffling
	
	number_ef_examples = data.shape[0]
	random_seq = range(0, number_ef_examples) 
	random.shuffle(random_seq)
	data = data[random_seq, :, :]
	labels = labels[random_seq, :]


	# Normalizing labels between 0 and 1

	max_val_0 = np.max(labels[:, 0])
	min_val_0 = np.min(labels[:, 0])
	labels[:, 0] = (labels[:, 0] - min_val_0)/(max_val_0 - min_val_0)

	max_val_1 = np.max(labels[:, 1])
	min_val_1 = np.min(labels[:, 1])
	labels[:, 1] = (labels[:, 1] - min_val_1)/(max_val_1 - min_val_1)

	max_val_2 = np.max(labels[:, 2])
	min_val_2 = np.min(labels[:, 2])
	labels[:, 2] = (labels[:, 2] - min_val_2)/(max_val_2 - min_val_2)
	
	max_val_3 = np.max(labels[:, 3])
	min_val_3 = np.min(labels[:, 3])
	labels[:, 3] = (labels[:, 3] - min_val_3)/(max_val_3 - min_val_3)

	
	# Spliting the dataset according to different signal modalities

	data_train = data[0:int(0.7*number_ef_examples), :, :]
	data_valid = data[int(0.7*number_ef_examples):int(0.9*number_ef_examples), :, :]
	data_test = data[int(0.9*number_ef_examples):-1, :, :]
	labels_train = labels[0:int(0.7*number_ef_examples), :]
	labels_valid = labels[int(0.7*number_ef_examples):int(0.9*number_ef_examples), :]
	labels_test = labels[int(0.9*number_ef_examples):-1, :]
	

	dataset_file_train = h5py.File(hdf_filename_to_save_train, 'w')
	dataset_train = dataset_file_train.create_dataset('data', data = data_train)
	dataset_train = dataset_file_train.create_dataset('labels', data = labels_train)
	dataset_file_train.close()

	dataset_file_valid = h5py.File(hdf_filename_to_save_valid, 'w')
	dataset_valid = dataset_file_valid.create_dataset('data', data = data_valid)
	dataset_valid = dataset_file_valid.create_dataset('labels', data = labels_valid)
	dataset_file_valid.close()

	dataset_file_test = h5py.File(hdf_filename_to_save_test, 'w')
	dataset_test = dataset_file_test.create_dataset('data', data = data_test)
	dataset_test = dataset_file_test.create_dataset('labels', data = labels_test)
	dataset_file_test.close()


def read_hdf(hdf_filename = 'DEAP_dataset_subjects_list.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data = open_file['data']
	labels = open_file['labels']

	return data, labels

def read_split_data_hdf(hdf_filename = 'DEAP_dataset.hdf'):

	open_file = h5py.File(hdf_filename, 'r')
	
	data_eeg = open_file['eeg']
	data_eog = open_file['eog']
	data_emg = open_file['emg']
	data_gsr = open_file['gsr']
	data_resp = open_file['resp']
	data_plet = open_file['plet']
	data_temp = open_file['temp']
	labels = open_file['labels']

	return data_eeg, data_eog, data_emg, data_gsr, data_resp, data_plet, data_temp, labels

def minibatch_generator_multimodal(dataset_filename = 'DEAP_dataset.hdf', dataset_size = 80640, minibatch_size = 32): #80640 = 32 sub x 63 segments/trial x 40 trials		
	
	number_of_slices = int(np.ceil(dataset_size/minibatch_size))
	
	while True:
		open_file = h5py.File(dataset_filename, 'r')

		for i in xrange(0, number_of_slices):
			eeg_minibatch = open_file['eeg'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			eog_minibatch = open_file['eog'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			emg_minibatch = open_file['emg'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			gsr_minibatch = open_file['gsr'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			resp_minibatch = open_file['resp'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			plet_minibatch = open_file['plet'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			temp_minibatch = open_file['temp'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			labels_minibatch = open_file['labels'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
       			
       			yield (eeg_minibatch, eog_minibatch, emg_minibatch, gsr_minibatch, resp_minibatch, plet_minibatch, temp_minibatch, labels_minibatch)

		open_file.close()

def minibatch_generator_train(dataset_filename = 'DEAP_dataset_train.hdf', dataset_size = 80640*0.7, minibatch_size = 32): #80640 = 32 sub x 63 segments/trial x 40 trials		
	
	number_of_slices = int(np.ceil(dataset_size/minibatch_size))
	
	while True:
		open_file = h5py.File(dataset_filename, 'r')

		for i in xrange(0, number_of_slices):
			data_minibatch_train = open_file['data'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			labels_minibatch_train = open_file['labels'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
       			
       			yield (data_minibatch_train, labels_minibatch_train)

		open_file.close()

def minibatch_generator_valid(dataset_filename = 'DEAP_dataset_valid.hdf', dataset_size = 80640*0.2, minibatch_size = 32): #80640 = 32 sub x 63 segments/trial x 40 trials		
	
	number_of_slices = int(np.ceil(dataset_size/minibatch_size))
	
	while True:
		open_file = h5py.File(dataset_filename, 'r')

		for i in xrange(0, number_of_slices):
			data_minibatch_valid = open_file['data'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
			labels_minibatch_valid = open_file['labels'][i*minibatch_size:min((i+1)*minibatch_size, dataset_size)]
       			
       			yield (data_minibatch_valid, labels_minibatch_valid)

		open_file.close()				

if __name__ == '__main__':

	#merge_shuffle_norm_split_tvt_store_as_hdf()

	#read_hdf('DEAP_dataset_train.hdf')
	#read_hdf('DEAP_dataset_valid.hdf')
	#read_hdf('DEAP_dataset_test.hdf')

	gen = minibatch_generator_valid()
	data, labels = gen.next()
	print(data.shape)
	print(labels.shape)

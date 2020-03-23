import os
import argparse
import torch
import numpy as np
from util import read_video
from tcn import define_model
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
IMAGE_SIZE = (299, 299)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/', help='Directory to save the models')
	parser.add_argument('--load-model', type=str, required=True, help='The model to be loaded')
	parser.add_argument('--validation-directory', type=str, default='./data/val/', help='Directory with the validation videos')
	parser.add_argument('--num-validation', type=int, default=10, help='Number of validation videos to use')
	parser.add_argument('--neighbours', type=int, default=5, help='Number of neighbours to use in KNN calculations')
	parser.add_argument('--test-video', type=str, required=True, help='Path to the reference video')
	return parser.parse_args()

arguments = get_args()

def model_filename(model_name, epoch):
	return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

def create_model(use_cuda):
	tcn = define_model(use_cuda)
	# tcn = PosNet()
	if arguments.load_model:
		model_path = os.path.join(
			arguments.model_folder,
			arguments.load_model
		)
		# map_location allows us to load models trained on cuda to cpu.
		tcn.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

	if use_cuda:
		tcn = tcn.cuda()
	return tcn


def main():
	with torch.no_grad():
		use_cuda = torch.cuda.is_available()
		print('CUDA Availability:',use_cuda)
		tcn = create_model(use_cuda)
		print('Created model')
		len_list = [0]
		
		validation_list = os.listdir(arguments.validation_directory)
		print('Got validation list from',arguments.validation_directory)
		validation_idx = np.random.choice(range(len(validation_list)), arguments.num_validation)
		validation_frames = []
		for idx in validation_idx:
			frames_list = read_video(os.path.join(arguments.validation_directory,validation_list[idx]), IMAGE_SIZE)
			validation_frames = validation_frames + frames_list.tolist()
			len_list.append(len(frames_list) + len_list[-1])
		print('Got',len(validation_frames),'frames from',arguments.num_validation,'validation videos')
		validation_frames = np.array(validation_frames)
		print(validation_frames.shape)

		validation_embeddings = []
		
		for i in range(int(len(validation_frames)/100)):
			embeddings = tcn(torch.Tensor(validation_frames[100*i:100*(i+1)]).cuda()).cpu().numpy().tolist()
			validation_embeddings = validation_embeddings + embeddings
		print('Got embeddings for validation frames')
		
		test_frames = read_video(arguments.test_video, IMAGE_SIZE)
		print('Got frames from test video',arguments.test_video)
		test_embeddings = tcn(torch.Tensor(test_frames).cuda()).cpu().numpy()
		print('Got embeddings for test frames')

		nn = NearestNeighbors(n_neighbors=arguments.neighbours).fit(validation_embeddings)
		print('Created NN object with validation embeddings')
		test_neighbours, test_distances = nn.kneighbors(test_embeddings, n_neighbors=1)
		print('Calculated neighbours for test embeddings')
		print(test_distances.mean())
		
		for i in range(len(test_frames)):
			test_frame = test_frames[i].transpose()
			val_frame = validation_frames[test_neighbours[i]][0].transpose()
			cv2.imwrite('%06d.png'%i, np.hstack([test_frame, val_frame]))

if __name__ == '__main__':
	main()

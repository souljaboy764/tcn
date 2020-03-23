import os
import argparse
import torch
import numpy as np
from torch import optim
import multiprocessing
from threading import Thread
from queue import Queue
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset
from util import (SingleViewTripletBuilder, MultiViewTripletBuilder, distance, Logger, ensure_folder)
from tcn import define_model

IMAGE_SIZE = (299, 299)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--start-epoch', type=int, default=1)
	parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to run the training for')
	parser.add_argument('--save-every', type=int, default=25, help='Number of epochs after which the model gets saved')
	parser.add_argument('--model-folder', type=str, default='./trained_models/tcn/', help='Directory to save the models')
	parser.add_argument('--load-model', type=str, required=False, help='The model to be loaded')
	parser.add_argument('--mode', choices=['single','multi'], default='multi', help='Whether to use Single View or Multi View TCN')
	parser.add_argument('--train-directory', type=str, default='./data/train/', help='Directory with the training videos')
	parser.add_argument('--validation-directory', type=str, default='./data/val/', help='Directory with the validation videos')
	# Using a suffix to differentiate the different views, like in the orinigal Tensorflow code.
	parser.add_argument('--train-suffix', type=str, default='_C[.]*.*', help='Suffix to partition training videos based on the different views')

	parser.add_argument('--minibatch-size', type=int, default=32, help='Mini batch size to use for training')
	parser.add_argument('--margin', type=float, default=2.0, help='Margin for the loss function')
	parser.add_argument('--model-name', type=str, default='tcn', help='Name for the model to be saved')
	parser.add_argument('--log-file', type=str, default='./out.log', help='Path to the log file')
	parser.add_argument('--lr-start', type=float, default=0.01, help='Initial learning rate')
	parser.add_argument('--triplets-from-videos', type=int, default=5)
	return parser.parse_args()

arguments = get_args()

logger = Logger(arguments.log_file)
def batch_size(epoch, max_size):
	exponent = epoch // 100
	return min(max(2 ** (exponent), 2), max_size)

if arguments.mode == 'single':
	validation_builder = SingleViewTripletBuilder(arguments.validation_directory, IMAGE_SIZE, arguments, sample_size=100)
elif arguments.mode == 'multi':
	validation_builder = MultiViewTripletBuilder(arguments.validation_directory, arguments.train_suffix, 3, IMAGE_SIZE, [], sample_size=100)
logger.info('Building validation sets')
validation_set = [validation_builder.build_set() for i in range(10)]
logger.info('Built validation sets')
logger.info('Concatenating validation sets')
validation_set = ConcatDataset(validation_set)
logger.info('Concatenated validation sets')

del validation_builder

def validate(tcn, use_cuda, arguments):
	with torch.no_grad():
		# Run model on validation data and log results
		data_loader = DataLoader(validation_set, batch_size=100, shuffle=False)
		correct_with_margin = 0
		correct_without_margin = 0
		for minibatch, _ in data_loader:
			frames = Variable(minibatch)

			if use_cuda:
				frames = frames.cuda()

			anchor_frames = frames[:, 0, :, :, :]
			positive_frames = frames[:, 1, :, :, :]
			negative_frames = frames[:, 2, :, :, :]

			anchor_output = tcn(anchor_frames)
			positive_output = tcn(positive_frames)
			negative_output = tcn(negative_frames)

			d_positive = distance(anchor_output, positive_output)
			d_negative = distance(anchor_output, negative_output)

			assert(d_positive.size()[0] == minibatch.size()[0])

			correct_with_margin += ((d_positive + arguments.margin) < d_negative).data.cpu().numpy().sum()
			correct_without_margin += (d_positive < d_negative).data.cpu().numpy().sum()

		message = "Validation score correct with margin {with_margin}/{total} and without margin {without_margin}/{total}".format(
			with_margin=correct_with_margin,
			without_margin=correct_without_margin,
			total=len(validation_set)
		)
		logger.info(message)

def model_filename(model_name, epoch):
	return "{model_name}-epoch-{epoch}.pk".format(model_name=model_name, epoch=epoch)

def save_model(model, filename, model_folder):
	ensure_folder(model_folder)
	model_path = os.path.join(model_folder, filename)
	torch.save(model.state_dict(), model_path)


def build_set(queue, triplet_builder, log):
	while 1:
		datasets = []
		for i in range(5):
			dataset = triplet_builder.build_set()
			datasets.append(dataset)
		dataset = ConcatDataset(datasets)
		log.info('Created {0} triplets'.format(len(dataset)))
		queue.put(dataset)

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
	use_cuda = torch.cuda.is_available()
	logger.info('CUDA status: {0}'.format(use_cuda))
	logger.info('Creating TCN Model')
	tcn = create_model(use_cuda)

	if arguments.mode=='single':
		triplet_builder = SingleViewTripletBuilder(arguments.train_directory, IMAGE_SIZE, arguments, sample_size=100)
	elif arguments.mode=='multi':
		triplet_builder = MultiViewTripletBuilder(arguments.train_directory, arguments.train_suffix, 3, IMAGE_SIZE, [], sample_size=100)

	queue = Queue(1)
	worker = Thread(target=build_set, args=(queue, triplet_builder, logger,))
	worker.setDaemon(True)
	worker.start()

	optimizer = optim.SGD(tcn.parameters(), lr=arguments.lr_start, momentum=0.9)
	# This will diminish the learning rate at the milestones.
	# 0.1, 0.01, 0.001
	learning_rate_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[100, 500, 1000], gamma=0.1)

	ITERATE_OVER_TRIPLETS = 5

	for epoch in range(arguments.start_epoch, arguments.start_epoch + arguments.epochs):
		logger.info("Starting epoch: {0} learning rate: {1}".format(epoch,
			learning_rate_scheduler.get_lr()))
		dataset = queue.get()
		logger.info("Got {0} triplets".format(len(dataset)))
		data_loader = DataLoader(
			dataset=dataset,
			batch_size=arguments.minibatch_size, # batch_size(epoch, arguments.max_minibatch_size),
			shuffle=True
		)
		if epoch % 10 == 0:
			logger.info('Validating after {0} epochs'.format(epoch))
			validate(tcn, use_cuda, arguments)

		for _ in range(0, ITERATE_OVER_TRIPLETS):
			losses = []
			for minibatch, _ in data_loader:
				frames = Variable(minibatch)

				if use_cuda:
					frames = frames.cuda()

				anchor_frames = frames[:, 0, :, :, :]
				positive_frames = frames[:, 1, :, :, :]
				negative_frames = frames[:, 2, :, :, :]

				anchor_output = tcn(anchor_frames)
				positive_output = tcn(positive_frames)
				negative_output = tcn(negative_frames)

				d_positive = distance(anchor_output, positive_output)
				d_negative = distance(anchor_output, negative_output)
				loss = torch.clamp(arguments.margin + d_positive - d_negative, min=0.0).mean()

				losses.append(loss.data.cpu().numpy())
				
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
		
			logger.info('loss: ', np.mean(losses))
		learning_rate_scheduler.step()
		if epoch % arguments.save_every == 0 and epoch != 0:
			logger.info('Saving model.')
			save_model(tcn, model_filename(arguments.model_name, epoch), arguments.model_folder)

	worker.join()



if __name__ == '__main__':
	main()

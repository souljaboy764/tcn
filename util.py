import glob
import cv2
import os
import functools
import imageio
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from torch import Tensor
from torch.autograd import Variable
import logging
import re

def distance(x1, x2):
	diff = torch.abs(x1 - x2)
	return torch.pow(diff, 2).sum(dim=1)

def view_image(frame):
	# For debugging. Shows the image
	# Input shape (3, 299, 299) float32
	img = Image.fromarray(np.transpose(frame * 255, [1, 2, 0]).astype(np.uint8))
	img.show()

def write_to_csv(values, keys, filepath):
	if  not(os.path.isfile(filepath)):
		with open(filepath, 'w', newline='') as csvfile:
			filewriter = csv.writer(csvfile)
			filewriter.writerow(keys)
			filewriter.writerow(values)
	else:
		with open(filepath, 'a', newline='') as csvfile:
			filewriter = csv.writer(csvfile)
			filewriter.writerow(values)


def ensure_folder(folder):
	path_fragments = os.path.split(folder)
	joined = '.'
	for fragment in path_fragments:
		joined = os.path.join(joined, fragment)
		if not os.path.exists(joined):
			os.mkdir(joined)

def _resize_frame(frame, out_size):
	image = Image.fromarray(frame)
	image = image.resize(out_size)
	scaled = np.array(image, dtype=np.float32) / 255
	return np.transpose(scaled, [2, 0, 1])

def write_video(file_name, path, frames):
	imageio.mimwrite(os.path.join(path, file_name), frames, fps=60)

def read_video(filepath, frame_size):
	if type(filepath)==str:
		files = [filepath]  
	elif type(filepath)==list: 
		files = filepath
	else: print(type(filepath))
	frame_list = []
	for f in files:
		cap = cv2.VideoCapture(f)
		total_frames = int(cap.get(7))
		frames = np.zeros((total_frames, 3, *frame_size))
		for i in range(total_frames):
			ret, frame = cap.read()
			frames[i, :, :, :] = cv2.resize(frame, frame_size).transpose()
		frame_list.append(frames)
		cap.release()
	if len(frame_list)==1:
		return frame_list[0]
	else:
		return frame_list

def ls_directories(path):
	return next(os.walk(path))[1]

def ls(path):
	# returns list of files in directory without hidden ones.
	return [p for p in os.listdir(path) if p[0] != '.']

class SingleViewTripletBuilder(object):
	def __init__(self, video_directory, image_size, cli_args, sample_size=500):
		self.frame_size = image_size
		self._read_video_dir(video_directory)

		self._count_frames()
		# The negative example has to be from outside the buffer window. Taken from both sides of
		# ihe frame.
		self.positive_frame_margin = 10
		self.negative_frame_margin = 30
		self.video_index = 0
		self.cli_args = cli_args
		self.sample_size = sample_size

	def _read_video_dir(self, video_directory):
		self._video_directory = video_directory
		filenames = ls(video_directory)
		self.video_paths = [os.path.join(self._video_directory, f) for f in filenames]
		self.video_count = len(self.video_paths)

	def _count_frames(self):
		frame_lengths = np.array([len(imageio.read(p)) for p in self.video_paths])
		self.frame_lengths = frame_lengths
		self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
		prev = 0
		for i, frames in enumerate(self.frame_lengths):
			prev = self.cumulative_lengths[i-1]
			self.cumulative_lengths[i] = prev + frames

	@functools.lru_cache(maxsize=1)
	def get_video(self, index):
		return read_video(self.video_paths[index], self.frame_size)

	def sample_triplet(self, snap):
		anchor_index = self.sample_anchor_frame_index()
		positive_index = self.sample_positive_frame_index(anchor_index)
		negative_index = self.sample_negative_frame_index(anchor_index)
		anchor_frame = snap[anchor_index]
		positive_frame = snap[positive_index]
		negative_frame = snap[negative_index]
		return (torch.Tensor(anchor_frame), torch.Tensor(positive_frame),
			torch.Tensor(negative_frame))

	def build_set(self):
		triplets = []
		triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
		for i in range(0, self.sample_size):
			snap = self.get_video(self.video_index)
			anchor_frame, positive_frame, negative_frame = self.sample_triplet(snap)
			triplets[i, 0, :, :, :] = anchor_frame
			triplets[i, 1, :, :, :] = positive_frame
			triplets[i, 2, :, :, :] = negative_frame

		self.video_index = (self.video_index + 1) % self.video_count
		# Second argument is labels. Not used.
		return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

	def sample_anchor_frame_index(self):
		arange = np.arange(0, self.frame_lengths[self.video_index])
		return np.random.choice(arange)

	def sample_positive_frame_index(self, anchor_index):
		lower_bound = max(0, anchor_index - self.positive_frame_margin)
		range1 = np.arange(lower_bound, anchor_index)
		upper_bound = min(self.frame_lengths[self.video_index] - 1, anchor_index + self.positive_frame_margin)
		range2 = np.arange(anchor_index + 1, upper_bound)
		return np.random.choice(np.concatenate([range1, range2]))

	def negative_frame_indices(self, anchor_index):
		video_length = self.frame_lengths[self.video_index]
		lower_bound = 0
		upper_bound = max(0, anchor_index - self.negative_frame_margin)
		range1 = np.arange(lower_bound, upper_bound)
		lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
		upper_bound = video_length
		range2 = np.arange(lower_bound, upper_bound)
		return np.concatenate([range1, range2])

	def sample_negative_frame_index(self, anchor_index):
		return np.random.choice(self.negative_frame_indices(anchor_index))

class Logger(object):
	def __init__(self, logfilename):
		logging.basicConfig(filename=logfilename, level=logging.DEBUG, filemode='a')

	def info(self, *arguments):
		print(*arguments)
		message = " ".join(map(repr, arguments))
		logging.info(message)

class MultiViewTripletBuilder(object):
	def __init__(self, video_directory, suffix_pattern, num_views, image_size, cli_args, sample_size=500):
		self.frame_size = image_size
		self.num_views = num_views
		self._suffix_pattern = suffix_pattern
		self._read_video_dir(video_directory)
		self.video_names = list(self.video_paths.keys())
		self._count_frames()
		# The negative example has to be from outside the buffer window. Taken from both sides of
		# ihe frame.
		self.positive_frame_margin = 10
		self.negative_frame_margin = 30
		self.video_index = 0
		self.cli_args = cli_args
		self.sample_size = sample_size

	def _read_video_dir(self, video_directory):
		self._video_directory = video_directory
		filenames = ls(video_directory)
		view_patt = re.compile('.*' + self._suffix_pattern)
		self.video_paths = {}
		for filename in filenames:
			if view_patt.match(filename):
				fullpath = re.sub(self._suffix_pattern, '', filename)
				num_views, min_num_frames, view_paths, num_frames = self._get_view_info(os.path.join(video_directory, fullpath+self._suffix_pattern[0]+'*'))
				if max(num_frames) - min(num_frames) > 100:
					continue
				if fullpath in self.video_paths:
					self.video_paths[fullpath].append(os.path.join(video_directory, filename))
				else:
					self.video_paths[fullpath] = [os.path.join(video_directory, filename)]
		self.video_count = len(self.video_paths.keys())
	
	# Taken from the original TCN code
	def _get_num_frames(self, vid_path):
		"""Gets the number of frames in a video."""
		cap = cv2.VideoCapture(vid_path)
		total_frames = cap.get(7)
		cap.release()
		return int(total_frames)

	# Taken from the original TCN code
	def _get_view_info(self, views_fullname):
		"""Return information about a group of views."""
		view_paths = sorted(glob.glob(views_fullname))
		num_frames = [self._get_num_frames(i) for i in view_paths]
		min_num_frames = min(num_frames)
		num_views = len(view_paths)
		return num_views, min_num_frames, view_paths, num_frames

	def _count_frames(self):
		np.array([self._get_num_frames(self.video_paths[p][0]) for p in self.video_paths])
		frame_lengths = []
		for p in self.video_paths:
			snap_lengths = []
			for snap in self.video_paths[p]:
				snap_lengths.append(self._get_num_frames(snap))
			frame_lengths.append(min(snap_lengths))
		
		self.frame_lengths = frame_lengths
		self.cumulative_lengths = np.zeros(len(self.frame_lengths), dtype=np.int32)
		prev = 0
		for i, frames in enumerate(self.frame_lengths):
			prev = self.cumulative_lengths[i-1]
			self.cumulative_lengths[i] = prev + frames

	@functools.lru_cache(maxsize=1)
	def get_video(self, index):
		return read_video(self.video_paths[self.video_names[index]], self.frame_size)
	
	def sample_triplet(self, snap):
		anchor_index = self.sample_anchor_frame_index()
		negative_index = self.sample_negative_frame_index(anchor_index)
		if negative_index==-1:
			return None

		anchor_frame = snap[0][anchor_index]
		negative_frame = snap[0][negative_index]
		triplets = []
		for i in range(1,len(snap)):
			triplets.append((torch.Tensor(anchor_frame), torch.Tensor(snap[i][anchor_index]),
			torch.Tensor(negative_frame)))
		return triplets

	def build_set(self):
		triplets = []
		batch_size = int(self.sample_size/(self.num_views-1))
		triplets = torch.Tensor(self.sample_size, 3, 3, *self.frame_size)
		count = 0
		for i in range(0, batch_size):
			snap = self.get_video(self.video_index)
			triplet_samples = self.sample_triplet(snap)
			if triplet_samples==None:
				continue
			for (anchor_frame, positive_frame, negative_frame)  in triplet_samples:
				triplets[count, 0, :, :, :] = anchor_frame
				triplets[count, 1, :, :, :] = positive_frame
				triplets[count, 2, :, :, :] = negative_frame
				count = count+1

		self.video_index = (self.video_index + 1) % self.video_count
		# Second argument is labels. Not used.
		return TensorDataset(triplets, torch.zeros(triplets.size()[0]))

	def sample_anchor_frame_index(self):
		arange = np.arange(0, self.frame_lengths[self.video_index])
		return np.random.choice(arange)

	def sample_positive_frame_index(self, anchor_index):
		lower_bound = max(0, anchor_index - self.positive_frame_margin)
		range1 = np.arange(lower_bound, anchor_index)
		upper_bound = min(self.frame_lengths[self.video_index] - 1, anchor_index + self.positive_frame_margin)
		range2 = np.arange(anchor_index + 1, upper_bound)
		return np.random.choice(np.concatenate([range1, range2]))

	def negative_frame_indices(self, anchor_index):
		video_length = self.frame_lengths[self.video_index]
		lower_bound = 0
		upper_bound = max(0, anchor_index - self.negative_frame_margin)
		range1 = np.arange(lower_bound, upper_bound)
		lower_bound = min(anchor_index + self.negative_frame_margin, video_length)
		upper_bound = video_length
		range2 = np.arange(lower_bound, upper_bound)
		return np.concatenate([range1, range2])

	def sample_negative_frame_index(self, anchor_index):
		try:
			return np.random.choice(self.negative_frame_indices(anchor_index))
		except:
			return -1
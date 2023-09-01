import os
import sys

import torch
import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF

import numpy as np
from PIL import Image
import glob
import random
import cv2
from utils import make_dataset, edge_compute
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
random.seed(1143)


def populate_train_list(hazy_images_path):


	train_list = []
	val_list = []
	test_list = []	
	image_list_haze = glob.glob(hazy_images_path + "*.png")


	'''tmp_dict = {}

	for image in image_list_haze:
		image = image.split("/")[-1]
		#key = image.split("_")[0] + "_" + image.split("_")[1] + ".jpg"
		key = image#.split("_")[1] #+ ".png"
		if key in tmp_dict.keys():
			tmp_dict[key].append(image)
		else:
			tmp_dict[key] = []
			tmp_dict[key].append(image)


	train_keys = []
	val_keys = []
	test_keys = []

	len_keys = len(tmp_dict.keys())
	for i in range(len_keys):
		if i < len_keys*9/10:
			train_keys.append(list(tmp_dict.keys())[i])
			test_keys.append(list(tmp_dict.keys())[i])
		else:
			val_keys.append(list(tmp_dict.keys())[i])
			test_keys.append(list(tmp_dict.keys())[i])


	for key in list(tmp_dict.keys()):

		if key in train_keys:
			for hazy_image in tmp_dict[key]:

				train_list.append([orig_images_path + key, hazy_images_path + hazy_image])
				test_list.append([orig_images_path + key, hazy_images_path + hazy_image])


		else:
			for hazy_image in tmp_dict[key]:

				val_list.append([orig_images_path + key, hazy_images_path + hazy_image])
				test_list.append([orig_images_path + key, hazy_images_path + hazy_image])



	random.shuffle(train_list)
	random.shuffle(val_list)
	random.shuffle(test_list)

	return train_list, val_list, test_list'''
	#print(image_list_haze)
	return image_list_haze




'''class dehazing_loader(data.Dataset):

	def __init__(self, orig_images_path, hazy_images_path, mode='train'):

		self.train_list, self.val_list, self.test_list = populate_train_list(orig_images_path, hazy_images_path) 

		if mode == 'train':
			self.data_list = self.train_list
			print("Total training examples:", len(self.train_list))
		elif mode=='val':
			self.data_list = self.val_list
			print("Total validation examples:", len(self.val_list))
		elif mode=='test':
			self.data_list = self.test_list
			print("Total test examples:", len(self.test_list))'''

class dehazing_loader(data.Dataset):

	def __init__(self, hazy_images_path, mode='train'):

		self.test_list = populate_train_list(hazy_images_path) 

		'''if mode == 'train':
			self.data_list = self.train_list
			print("Total training examples:", len(self.train_list))
		elif mode=='val':
			self.data_list = self.val_list
			print("Total validation examples:", len(self.val_list))'''
		if mode=='test':
			self.data_list = self.test_list
			print("Total test examples:", len(self.test_list))

	def __getitem__(self, index):

		data_hazy_path = self.data_list[index]
		#image_name = data_hazy_path.split("/")[-1]
		#print(image_name)

		#data_orig = Image.open(data_orig_path)
		data_hazy = Image.open(data_hazy_path)
		#if data_orig.shape[0] == 4:
		#data_orig = data_orig.convert("RGB") 
		#if data_hazy.shape[0] == 4:
		data_hazy = data_hazy.convert("RGB")
		#data_orig=tfs.CenterCrop(data_hazy.size[::-1])(data_orig)
		#data_orig=tfs.CenterCrop((240,240))(data_orig)
		#data_hazy=tfs.CenterCrop((240,240))(data_hazy)
		'''i,j,h,w=tfs.RandomCrop.get_params(data_hazy,output_size=(240,240))
		data_hazy=FF.crop(data_hazy,i,j,h,w)
		data_orig=FF.crop(data_orig,i,j,h,w)'''
		#data_hazy=FF.crop(data_hazy,i,j,h,w)
		#data_orig=FF.crop(data_orig,i,j,h,w)
		#data_orig = data_orig.resize((480,640), Image.ANTIALIAS)
		#data_hazy = data_hazy.resize((480,640), Image.ANTIALIAS)
		#print(data_hazy.shape)
		#data_orig = (np.asarray(data_orig)/255.0) 
		data_hazy = (np.asarray(data_hazy)/255.0) 
		#print(data_hazy.shape)
		#data_orig = torch.from_numpy(data_orig.transpose((2, 0, 1))).float()
		data_hazy = torch.from_numpy(data_hazy.transpose((2, 0, 1))).float()
		#edge_data = edge_compute(data_hazy)
		#in_data = torch.cat((data_hazy, edge_data), dim=0)#.unsqueeze(0) - 128
		#return data_orig, data_hazy#, in_data
		return data_hazy_path, data_hazy#, in_data

	def __len__(self):
		return len(self.data_list)


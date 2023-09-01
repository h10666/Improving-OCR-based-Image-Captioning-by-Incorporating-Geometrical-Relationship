import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import test_dataloader1
#import net
import numpy as np
from torchvision import transforms
#from GCANet import GCANet
from torch.autograd import Variable
from PIL import Image
#from GCANetPSConv_new import GCANetPSConv
from PSConv import PSConv
from MS_SSIM import MS_SSIM
import pytorch_ssim
from math import log10
from os.path import join
import warnings
warnings.filterwarnings("ignore",category=UserWarning)

torch.cuda.set_device(1)

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m,nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)    
    elif isinstance(m,nn.InstanceNorm2d):
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


def train(config):

	dehaze_net = PSConv(in_c=3, out_c=3).cuda()
	dehaze_net = torch.nn.DataParallel((dehaze_net), device_ids=[1,2,3])
	optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, betas=(0.9,0.999), weight_decay=config.weight_decay)
	#optimizer = nn.DataParallel(optimizer)#,device_ids=[0,1])

	if config.pretrained:
		print("pretrained model loading...")
		checkpoint = torch.load('snapshots/Epoch14.pth')
		dehaze_net.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		start_epoch = checkpoint['epoch'] + 1

	else:
		print("weights initing...")
		start_epoch = 0
		dehaze_net.apply(weights_init)

	test_dataset = test_dataloader1.dehazing_loader(config.test_orig_images_path,config.test_hazy_images_path, mode="test")
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, pin_memory=False)

	train_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path)		
	#val_dataset = dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="val")		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	#val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	criterion = nn.MSELoss().cuda()
	smooth_l1Loss = nn.SmoothL1Loss().cuda()
	tvLoss = TVLoss().cuda()
	msLoss = MS_SSIM().cuda()
	dehaze_net.train()
	max_ssim = 0 #0.9550
	max_psnr = 0 #27.79
	i = 0
	for epoch in range(start_epoch, start_epoch+config.num_epochs):
		#max_ssim = 0.9545
		#max_psnr = 27.72
		i = i + 1
		for iteration, (img_orig, img_haze) in enumerate(train_loader):
			dehaze_net.train()
			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()
			#in_data = in_data.cuda(0)

			clean_image = dehaze_net(img_haze)

			loss_M = criterion(clean_image, img_orig)
			loss_L1 = smooth_l1Loss(clean_image, img_orig)
			loss_tv = tvLoss(clean_image)
			loss_ms = msLoss(clean_image, img_orig)
			loss = loss_M * 150 + loss_ms * 150 * 0.01  + loss_L1 * 100 #+ loss_tv * 0.001
			#loss = loss_M * 150 + loss_ms * 150 * 0.01  + loss_L1 * 100 *0.1 #+ loss_tv * 0.001
			#loss = loss_M * 0.3 + loss_ms * 0.7

			optimizer.zero_grad()
			#loss.requires_grad = True      
			loss.backward()
			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)
			optimizer.step()
			#optimizer.module.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("************************************************")
				print("Loss at iteration", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				#dehaze_net.eval()
				criterion_test = nn.MSELoss().cuda()
				with torch.no_grad():
					ssim_eval, psnr_eval = test(dehaze_net, criterion_test, test_loader)
					if ssim_eval > max_ssim and psnr_eval > max_psnr:
						max_ssim = max(max_ssim,ssim_eval)
						max_psnr = max(max_psnr,psnr_eval)           
						state = {'model':dehaze_net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch}
						torch.save(state, config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
						#print("====================================================")
						print("===> Avg. SR MAX_SSIM: {:.4f} ".format(max_ssim))
						print("Avg. SR MAX_PSNR:{:4f} dB".format(max_psnr))
						state = {'model':dehaze_net.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch}
						torch.save(state, config.snapshots_folder + "dehazer.pth")
		print("==============================================================>{}".format(i))
def test(net, criterion, test_loader):
	avg_psnr = 0
	avg_ssim = 0
	#med_time = []
	net.eval()
	i = 0
	for iteration, (img_orig, img_haze) in enumerate(test_loader):
		img_orig = img_orig.cuda()
		img_haze = img_haze.cuda()
		#in_data = in_data.cuda()
		clean_image = net(img_haze)
		ssim = pytorch_ssim.ssim(clean_image,img_orig).cuda()
      
		avg_ssim += ssim
      
		mse = criterion(clean_image,img_orig)
		psnr = 10 * log10(1 / mse)
		i = i + 1
		#print(i)
		avg_psnr += psnr
		#print(iteration)
	avg_ssim = avg_ssim / i
	avg_psnr = avg_psnr / i
	print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim))
	print("Avg. SR PSNR:{:4f} dB".format(avg_psnr))
	return avg_ssim, avg_psnr

if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default="/home/sunlexuan/DLModel/Data/RESIDE/OTS/clear/")
	parser.add_argument('--hazy_images_path', type=str, default="/home/sunlexuan/DLModel/Data/RESIDE/OTS/hazy/")
	parser.add_argument('--test_orig_images_path', type=str, default="/home/sunlexuan/DLModel/Data/RESIDE/SOTS/outdoor/clear/")
	parser.add_argument('--test_hazy_images_path', type=str, default="/home/sunlexuan/DLModel/Data/RESIDE/SOTS/outdoor/hazy/")
	parser.add_argument('--lr', type=float, default=0.0002)
	parser.add_argument('--weight_decay', type=float, default=0)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=50)
	parser.add_argument('--train_batch_size', type=int, default=4)
	parser.add_argument('--val_batch_size', type=int, default=1)
	parser.add_argument('--test_batch_size', type=int, default=1)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=540)
	#parser.add_argument('--display_iter', type=int, default=100)
	parser.add_argument('--snapshot_iter', type=int, default=1080)
	#parser.add_argument('--snapshot_iter', type=int, default=50)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--sample_output_folder', type=str, default="samples/")
	parser.add_argument('--pretrained',type=bool,default=False)

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	#if not os.path.exists(config.sample_output_folder):
		#os.mkdir(config.sample_output_folder)

	train(config)
	torch.cuda.empty_cache()
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import test_dataloader
#import net
#from GCANet import GCANet
#from GCANetPSConv_new import GCANetPSConv
from PSConv import PSConv
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_ssim
from math import log10
from os.path import join
#from skimage.measure import compare_ssim
import warnings
warnings.filterwarnings("ignore",category=UserWarning)

def test(config):
  avg_psnr = 0
  avg_ssim = 0
  med_time = []
  i = 0
  test_dataset = test_dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path, mode="test")
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, pin_memory=False)
  print("===> Loading model and criterion")
  #dehaze_net = net.dehaze_net().cuda()
  dehaze_net = PSConv(in_c=3).cuda()
  dehaze_net = torch.nn.DataParallel(dehaze_net) 
 
  #dehaze_net.load_state_dict(torch.load('snapshots_PSconv_L1Se/Epoch35.pth',map_location=lambda storage, loc: storage))
  checkpoint = torch.load('snapshots/Epoch13.pth', map_location={'cuda:1':'cuda:0'})
  dehaze_net.load_state_dict(checkpoint['model'])
  dehaze_net.eval()
  criterion = nn.MSELoss().cuda()
  with torch.no_grad():
    for iteration, (img_orig, img_haze) in enumerate(test_loader):
      img_orig = img_orig.cuda()
      img_haze = img_haze.cuda()
      #in_data = in_data.cuda()
      clean_image = dehaze_net(img_haze)
      
      '''try:
        clean_image = torch.clamp(clean_image,min=0,max=1)
      except:
        clean_image = clean_image[0]
        clean_image = torch.clamp(clean_image,min=0,max=1)'''
      #torch.cuda.synchronize()
      
      ssim = pytorch_ssim.ssim(clean_image,img_orig).cuda()
      
      avg_ssim += ssim
      
      mse = criterion(clean_image,img_orig)
      psnr = 10 * log10(1 / mse)
      
      '''resultSRDeblur = transforms.ToPILImage()(clean_image.cpu()[0])
      i = i + 1
      resultSRDeblur.save(join('result_1','{0}.jpg'.format(i)))'''
      i = i + 1
      print(i)
      avg_psnr += psnr
    #print(iteration)
    print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim / i))
    print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / i))
    
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument('--orig_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/new_model/gt/")
  parser.add_argument('--hazy_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/new_model/test/")
  #parser.add_argument('--orig_images_path', type=str, default="/sunlexuan/DLModel/my_dehaze/HSTS/synthetic/original/")
  #parser.add_argument('--hazy_images_path', type=str, default="/sunlexuan/DLModel/my_dehaze/HSTS/synthetic/synthetic/")
  parser.add_argument('--test_batch_size', type=int, default=1)
  config = parser.parse_args()

  test(config)    
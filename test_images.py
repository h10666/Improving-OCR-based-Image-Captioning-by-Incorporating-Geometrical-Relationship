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
import numpy as np
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import pytorch_ssim
from math import log10
from os.path import join
from PSConv import PSConv
#from GCANet import GCANet
#from skimage.measure import compare_ssim
#torch.cuda.set_device(1)
'''def ssim(imageA, imageB):
  imageA = np.array(imageA, dtype=np.uint8)
  imageB = np.array(imageB, dtype=np.uint8)

  (B1, G1, R1) = cv2.split(imageA)
  (B2, G2, R2) = cv2.split(imageB)

  #convert the images to grayscale BGR2GRAY
  grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
  grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
  #(grayScore, diff) = compare_ssim(grayA, grayB, full=True)
  #diff = (diff * 255).astype("uint8")
  #print("gray SSIM: {}".format(grayScore))

  (score0, diffB) = compare_ssim(B1, B2, full=True)
  (score1, diffG) = compare_ssim(G1, G2, full=True)
  (score2, diffR) = compare_ssim(R1, R2, full=True)
  aveScore = (score0+score1+score2)/3
  print("BGR average SSIM: {}".format(aveScore))
	
  #return grayScore, aveScore
  return aveScore'''

def test(config):
  avg_psnr = 0
  avg_ssim = 0
  med_time = []
  i = 0
  #test_dataset = test_dataloader.dehazing_loader(config.orig_images_path, config.hazy_images_path, mode="test")
  test_dataset = test_dataloader.dehazing_loader(config.hazy_images_path, mode="test")
  test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False, pin_memory=False)
  print("===> Loading model and criterion")
  #dehaze_net = net.dehaze_net().cuda()
  dehaze_net = PSConv(in_c=3, out_c=3).cuda(0)
  dehaze_net = torch.nn.DataParallel(dehaze_net)#,device_ids=[1,2])
  #dehaze_net.load_state_dict(torch.load('snapshots/Epoch36.pth',map_location=lambda storage, loc: storage))
  checkpoint = torch.load('snapshots/Epoch14.pth')
  dehaze_net.load_state_dict(checkpoint['model'])
  #dehaze_net.load_state_dict(torch.load('snapshots/Epoch36.pth', map_location={'cuda:0':'cuda:1','cuda:3':'cuda:1'}))
  dehaze_net.eval()
  criterion = nn.MSELoss().cuda()
  with torch.no_grad():
    for iteration, (data_hazy_path, img_haze) in enumerate(test_loader):
      #img_orig = img_orig.cuda()
      data_hazy_path = "".join(tuple(data_hazy_path))
      image_name = data_hazy_path.split("/")[-1]
      new_path = config.sample_output_folder+image_name
      #if os.path.exists(new_path):
        #continue
      img_haze = img_haze.cuda()
      #in_data = in_data.cuda(1)

      #image_name = data_hazy_path.split("/")[-1]
      print("{0} is {1}".format(image_name, iteration+1))      
      clean_image = dehaze_net(img_haze)
      
      '''try:
        clean_image = torch.clamp(clean_image,min=0,max=1)
      except:
        clean_image = clean_image[0]
        clean_image = torch.clamp(clean_image,min=0,max=1)'''
      #torch.cuda.synchronize()
      
      '''ssim = pytorch_ssim.ssim(clean_image,img_orig).cuda()
      
      avg_ssim += ssim
      
      mse = criterion(clean_image,img_orig)
      psnr = 10 * log10(1 / mse)
      
      #resultSRDeblur = transforms.ToPILImage()(clean_image.cpu()[0])
      #i = i + 1
      #resultSRDeblur.save(join('result_1','{0}.jpg'.format(i)))
      i = i + 1
      print(i)
      avg_psnr += psnr
    #print(iteration)
    print("===> Avg. SR SSIM: {:.4f} ".format(avg_ssim / i))
    print("Avg. SR PSNR:{:4f} dB".format(avg_psnr / i))'''

      torchvision.utils.save_image(clean_image, config.sample_output_folder+image_name)
      #torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")
    
if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  #parser.add_argument('--orig_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/HSTS/synthetic/original/")
  parser.add_argument('--hazy_images_path', type=str, default="/home/sunlexuan/DLModel/my_model/spatial_test/")
  #parser.add_argument('--hazy_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/HSTS/real-world/")
  #parser.add_argument('--orig_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/HSTS/synthetic/original/")
  #parser.add_argument('--hazy_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/HSTS/synthetic/synthetic/")
  #parser.add_argument('--orig_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/indoor_gt/")
  #parser.add_argument('--hazy_images_path', type=str, default="/home/sunlexuan/DLModel/AOD_PyTorch-Image-Dehazing/indoor_test/")
  '''parser.add_argument('--lr', type=float, default=0.0001)
  parser.add_argument('--weight_decay', type=float, default=0.0001)
  parser.add_argument('--grad_clip_norm', type=float, default=0.1)
  parser.add_argument('--num_epochs', type=int, default=10)
  parser.add_argument('--train_batch_size', type=int, default=4)
  parser.add_argument('--val_batch_size', type=int, default=2)
  parser.add_argument('--num_workers', type=int, default=4)
  parser.add_argument('--display_iter', type=int, default=540)
  parser.add_argument('--snapshot_iter', type=int, default=180)
  parser.add_argument('--snapshots_folder', type=str, default="snapshots_1/")'''
  parser.add_argument('--sample_output_folder', type=str, default="/home/sunlexuan/DLModel/my_model/spatial_results/")
  parser.add_argument('--test_batch_size', type=int, default=1)
  config = parser.parse_args()

  '''if not os.path.exists(config.snapshots_folder):
    os.mkdir(config.snapshots_folder)
  if not os.path.exists(config.sample_output_folder):
    os.mkdir(config.sample_output_folder)'''

  test(config)    
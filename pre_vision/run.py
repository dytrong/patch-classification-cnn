import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from forward import *
import time
from compute_distance import *
#####global variable######
device=torch.device('cuda:1') #调用gpu:1
Model_Img_size=224
mini_batch_size=8
img_suffix='.jpg'
txt_suffix='.h5'
#####download models######
start=time.time()
myresnet=models.resnet18(pretrained=True).cuda(device)
#print(myresnet)
myresnet.eval() #不加这行代码，程序预测结果错误:
end=time.time()
print('init spend time '+str(end-start))
img_to_tensor = transforms.ToTensor()

class generate_des:
    def __init__(self,net,H5_Patch,Model_Img_size,mini_batch_size):
        self.img_tensor=self.change_images_to_tensor(H5_Patch,Model_Img_size)
        self.descriptor=self.extract_batch_conv_features(net,self.img_tensor.cuda(device),mini_batch_size)
    #####extract batch conv features#####
    def extract_batch_conv_features(self,net,input_data,mini_batch_size):
        batch_number=int(len(input_data)/mini_batch_size)
        descriptor_init=self.extract_conv_features(net,input_data[:mini_batch_size]).cpu().detach().numpy()
        for i in range(1,batch_number):
            mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            temp_descriptor=self.extract_conv_features(net,mini_batch).cpu().detach().numpy()
            descriptor_init=np.vstack((descriptor_init,temp_descriptor))
        #####avoid the last mini_batch is NULL######
        if (len(input_data)%mini_batch_size==0):
            return descriptor_init 
        descriptor=self.extract_conv_features(net,input_data[mini_batch_size*batch_number:len(input_data)+1]).cpu().detach().numpy()
        #####aviod the batch_number=0######
        if batch_number > 0:
            descriptor=np.vstack((descriptor_init,descriptor))
        return descriptor
    #####extract conv features#####
    def extract_conv_features(self,net,input_data,net_type='resnet'):
        if net_type.startswith('vgg16'):
            x=vgg16(net,input_data)  ####vgg16 is in forward.py
        if net_type.startswith('resnet'):
            x=resnet(net,input_data)
        if net_type.startswith('densenet'):
            x=densenet(net,input_data)
        if net_type.startswith('inception_v3'):
            x=inception_v3(net,input_data)
        return x    
    #####change images to tensor#####
    def change_images_to_tensor(self,H5_Patch,Model_Img_size):
        img_list=[]
        for i in range(1000):
            _H5_path=H5_Patch+str(i)+img_suffix
            img=Image.open(_H5_path)
            img=img.resize((Model_Img_size,Model_Img_size))
            img=img_to_tensor(img)
            img=img.numpy()
            img=img.reshape(3,Model_Img_size,Model_Img_size)
            img_list.append(img)
        img_array=np.array(img_list)
        img_tensor=torch.from_numpy(img_array)
        return img_tensor


if __name__=="__main__":
    H5_Patch_A='./AY/'
    H5_Patch_B='./BY/'
    desc1=generate_des(myresnet,H5_Patch_A,Model_Img_size,mini_batch_size).descriptor
    desc2=generate_des(myresnet,H5_Patch_B,Model_Img_size,mini_batch_size).descriptor
    MAX_XY=compute_max_match_cc(desc1,desc2)
    print(MAX_XY)
    print("一共检查出 "+str(len(MAX_XY)) +" 对匹配")
    count=0
    match_txt_path='./yosemite_dataset/m50_1000_1000_0.txt'
    ground_match_array=np.loadtxt(match_txt_path)
    for i in range(len(MAX_XY)):
        import cv2
        sub_path1='./AY/'+str(int(MAX_XY[i][0]))+'.jpg'
        sub_path2='./BY/'+str(int(MAX_XY[i][1]))+'.jpg'
        img1=cv2.imread(sub_path1)
        img2=cv2.imread(sub_path2)
        true_index1=int(MAX_XY[i][0]) ###图片在ground_match_array中的索引
        true_index2=int(MAX_XY[i][1])
        if ground_match_array[true_index1][1]==ground_match_array[true_index2][4]:
            count=count+1
            cv2.imshow('img1',img1)
            cv2.imshow('img2',img2)
            #cv2.waitKey(0)
        else:
            cv2.imshow('img1',img1)
            cv2.imshow('img2',img2)
            cv2.waitKey(0)
    print('其中正确的个数:',str(count))

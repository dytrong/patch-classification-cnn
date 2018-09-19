import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from forward import *
import time
from ROC import plot_roc
from compute_distance_v2 import *
#####global variable######
device=torch.device('cuda:0') #调用gpu:1
img_suffix='.jpg'
txt_suffix='.h5'
#####download models######
start=time.time()
myresnet=models.resnet50(pretrained=True).cuda(device)
#print(myresnet)
myresnet.eval() #不加这行代码，程序预测结果错误:
end=time.time()
print('init spend time '+str(end-start))
img_to_tensor = transforms.ToTensor()

class generate_des:
    def __init__(self,net,Img_path,Img_number,Model_Img_size=224,mini_batch_size=10,net_type='resnet'):
        self.img_tensor=self.change_images_to_tensor(Img_path,Model_Img_size,Img_number)
        self.descriptor=self.extract_batch_conv_features(net,self.img_tensor.cuda(device),mini_batch_size,net_type)
    #####extract batch conv features#####
    def extract_batch_conv_features(self,net,input_data,mini_batch_size,net_type):
        batch_number=int(len(input_data)/mini_batch_size)
        descriptor_init=self.extract_conv_features(net,input_data[:mini_batch_size],net_type).cpu().detach().numpy()
        start=time.time()
        for i in range(1,batch_number):
            mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            temp_descriptor=self.extract_conv_features(net,mini_batch,net_type).cpu().detach().numpy()
            descriptor_init=np.vstack((descriptor_init,temp_descriptor))
        end=time.time()
        print('加载数据耗时:'+str(end-start))
        #####avoid the last mini_batch is NULL######
        if (len(input_data)%mini_batch_size==0):
            return descriptor_init 
        descriptor=self.extract_conv_features(net,input_data[mini_batch_size*batch_number:len(input_data)+1],net_type).cpu().detach().numpy()
        #####aviod the batch_number=0######
        if batch_number > 0:
            descriptor=np.vstack((descriptor_init,descriptor))
        return descriptor
    #####extract conv features#####
    def extract_conv_features(self,net,input_data,net_type):
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
    def change_images_to_tensor(self,Img_path,Model_Img_size,Img_number):
        img_list=[]
        start=time.time()
        for i in range(Img_number):
            _Img_path=Img_path+str(i)+img_suffix
            img=Image.open(_Img_path)
            img=img.resize((Model_Img_size,Model_Img_size))
            img=img_to_tensor(img)
            img=img.numpy()
            img=img.reshape(3,Model_Img_size,Model_Img_size)
            img_list.append(img)
        img_array=np.array(img_list)
        img_tensor=torch.from_numpy(img_array)
        end=time.time()
        print('读取图片耗时:'+str(end-start))
        return img_tensor

def plot_hist(arr):
    import matplotlib.pyplot as plt
    #plt.figure("hist")
    #arr=arr.flatten()
    #n, bins, patches = plt.hist(arr, bins=len(arr), normed=1,edgecolor='None',facecolor='red')  
    x=np.arange(10000)
    plt.scatter(x,arr)
    plt.show()

if __name__=="__main__":
    Img_path_A='./AY/'
    Img_path_B='./BY/'
    Img_number=10000
    desc1=generate_des(myresnet,Img_path_A,Img_number).descriptor
    desc2=generate_des(myresnet,Img_path_B,Img_number).descriptor
    start=time.time()
    cos_dis=compute_L2_dis(desc1,desc2)
    #plot_hist(cos_dis)
    end=time.time()
    print('计算欧式距离共花费时间:'+str(end-start))
    median_number=1.6*np.median(cos_dis)
    print('欧式距离的中值:'+str(median_number))
    match_number=5000
    match_txt_path='./yosemite_dataset/m50_10000_10000_0.txt'
    for i in range(50):
        thresh=median_number-int(median_number/50)*i
        P,R=plot_roc(match_txt_path,cos_dis,thresh,match_number)
        print("精度为:",str(P))
        print("召回率:",str(R))
        print('\n')

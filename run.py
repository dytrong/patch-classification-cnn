import torch.nn as nn
import torchvision.models as models  
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from forward import *
import time
from ROC import plot_roc
from compute_distance import *
from plot_position_negative import compute_pos_neg_sample
#####global variable######
device=torch.device('cuda:1') #调用gpu:1
img_suffix='.jpg'
txt_suffix='.h5'
Model_Img_size=224
img_to_tensor = transforms.ToTensor()
#####download models######
start=time.time()
mynet=models.alexnet(pretrained=True).cuda(device)
print(mynet)
mynet.eval() #不加这行代码，程序预测结果错误:
end=time.time()
print('init spend time '+str(end-start))
######class#########
class generate_des:
    def __init__(self,net,img_tensor,mini_batch_size=8,net_type='alexnet'):
        self.descriptor=self.extract_batch_conv_features(net,img_tensor,mini_batch_size,net_type)
    #####extract batch conv features#####
    def extract_batch_conv_features(self,net,input_data,mini_batch_size,net_type):
        batch_number=int(len(input_data)/mini_batch_size)
        descriptor_init=self.extract_conv_features(net,input_data[:mini_batch_size],net_type).cpu().detach().numpy()
        #start=time.time()
        for i in range(1,batch_number):
            mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
            temp_descriptor=self.extract_conv_features(net,mini_batch,net_type).cpu().detach().numpy()
            descriptor_init=np.vstack((descriptor_init,temp_descriptor))
        #end=time.time()
        #print('加载数据耗时:'+str(end-start))
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
        if net_type.startswith('alexnet'):####alexnet is in forward.py
            x=alexnet(net,input_data)
        if net_type.startswith('vgg16'):
            x=vgg16(net,input_data) 
        if net_type.startswith('vgg19'):
            x=vgg19(net,input_data) 
        if net_type.startswith('inception_v3'):
            x=inception_v3(net,input_data)
        if net_type.startswith('squeezenet1'):
            x=squeezenet1(net,input_data) 
        if net_type.startswith('resnet'):
            x=resnet(net,input_data)
        if net_type.startswith('densenet'):
            x=densenet(net,input_data)
        return x    

#####read image########## 
def read_image(Img_path,Model_Img_size):
    img=Image.open(Img_path)
    img=img.resize((Model_Img_size,Model_Img_size))
    img=img_to_tensor(img)
    img=img.numpy()
    img=img.reshape(3,Model_Img_size,Model_Img_size)
    return img
#####change images to tensor#####
def change_images_to_tensor(Img_path,Model_Img_size,Img_number):
    img_list=[]
    start=time.time()
    for i in range(Img_number):
        _Img_path=Img_path+str(i)+img_suffix ###读入图像的地址 
        img=read_image(_Img_path,Model_Img_size) ####将图片数据转化为pytorch中预训练模型的输入格式
        img_list.append(img)
    img_array=np.array(img_list) ####将图片list,转化为numpy.array格式
    img_tensor=torch.from_numpy(img_array) #####将numpy.array格式转化为pytorch的Tensor数据格式
    end=time.time()
    print('读取图片耗时:'+str(end-start))
    return img_tensor

#####分批进行计算.如果一起读入，数据太大，读取时间太长，不能充分利用GPU.
def compute_batch_descriptor(net,input_data,mini_batch_size):
    batch_number=int(len(input_data)/mini_batch_size)  #####计算将数据分成了多少块
    descriptor_init=generate_des(mynet,input_data[:mini_batch_size].cuda(device)).descriptor ####第一个mini_batch
    start=time.time()
    ######循环读入mini_batch#######
    for i in range(1,batch_number):
        mini_batch=input_data[mini_batch_size*i:mini_batch_size*(i+1)]
        temp_descriptor=generate_des(mynet,mini_batch.cuda(device)).descriptor
        descriptor_init=np.vstack((descriptor_init,temp_descriptor)) ####将描述符叠起来
    end=time.time()
    print("计算描述符共耗时:"+str(end-start))
    #####avoid the last mini_batch is NULL######
    if (len(input_data)%mini_batch_size==0):
        return descriptor_init
    #####计算最后一块，可能为0，数据块小于mini_batch的大小
    descriptor=generate_des(net,input_data[mini_batch_size*batch_number:len(input_data)+1].cuda(device)).descriptor
    #####aviod the batch_number=0######
    if batch_number > 0:
        descriptor=np.vstack((descriptor_init,descriptor))
    return descriptor    

if __name__=="__main__":
    Img_path_A='./AY/'
    Img_path_B='./BY/'
    Img_number=10000
    ######读入图片数据，并转化为Tensor
    img_tensor_A=change_images_to_tensor(Img_path_A,Model_Img_size,Img_number)
    img_tensor_B=change_images_to_tensor(Img_path_B,Model_Img_size,Img_number)
    desc1=compute_batch_descriptor(mynet,img_tensor_A,64)
    desc2=compute_batch_descriptor(mynet,img_tensor_B,64)
    #print('desc2的形状:'+str(desc2.shape))
    #rank_2=np.linalg.matrix_rank(desc2)
    #print('desc2的秩为:'+str(rank_2))
    start=time.time()
    cos_dis=compute_L2_dis(desc1,desc2)
    cos_dis=cos_dis.cpu().numpy() ####将gpu数据转为cpu的numpy.array型数据
    end=time.time()
    print('计算欧式距离共花费时间:'+str(end-start))
    #########计算描述符距离的最大值，最小值,均值
    max_number=np.max(cos_dis)
    min_number=np.min(cos_dis)
    print('欧式距离的最大值:'+str(max_number))
    print('欧式距离的最小值:'+str(min_number))
    match_number=5000
    match_txt_path='/home/data1/daizhuang/patch_dataset/liberty_dataset/m50_10000_10000_0.txt'
    #compute_pos_neg_sample(match_txt_path,cos_dis)
    step_number=50
    P_list=[]
    R_list=[]
    AP=[]
    for i in range(step_number):
        thresh=max_number-((max_number-min_number)/step_number)*i
        P,R=plot_roc(match_txt_path,cos_dis,thresh,match_number)
        P_list.append(P)
        R_list.append(R)
    P_list.reverse() ###将列表反转
    R_list.reverse()
    print("精度为:",str(P_list))
    print("召回率:",str(R_list))
    AP.append(P_list[0]*R_list[0])
    for i in range(1,step_number):
        AP.append(P_list[i]*(R_list[i]-R_list[i-1])) 
    mAP=np.sum(AP)
    print("数据的mAP为:"+str(mAP))

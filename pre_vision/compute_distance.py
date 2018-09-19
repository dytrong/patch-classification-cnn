import cv2
import numpy as np
import h5py
import math
import time
from matplotlib import pyplot as plt
import torch
device=torch.device('cuda:1')
#################################
###des1:des1 is a numpy.array####
###des2:des2 is a numpy.array####
def compute_des_cos_dis(des1,des2):
    des1=torch.from_numpy(des1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    des2=torch.from_numpy(des2).cuda(device)
    des1_T=torch.transpose(des1,0,1) #### 转置
    des2_T=torch.transpose(des2,0,1) 
    temp_1=torch.mm(des1,des2_T).cpu().numpy() #### torch.mm 矩阵的点乘 ，data.cpu().numpy() 将gpu的tensor 转化为cpu的numpy
    temp_2=torch.pow(torch.mm(des1,des1_T),0.5).cpu().numpy()
    temp_3=torch.pow(torch.mm(des2,des2_T),0.5).cpu().numpy()
    temp_matrix=np.zeros((temp_2.shape[0],temp_3.shape[0])) #####初始化矩阵
    for i in range(temp_2.shape[0]):
        for j in range(temp_3.shape[0]):
            temp_matrix[i,j]=temp_2[i,i]*temp_3[j,j] ####取对角线元素相乘
    cos_dis_matrix=temp_1/temp_matrix     #####cosine distance
    return cos_dis_matrix

def compute_max_match_nn(des1,des2):
    cos_dis=compute_des_cos_dis(des1,des2)
    print("###compute the max match###")
    iterate_left=cos_dis.shape[0]
    Max_Matches=np.dtype({'names':['MAX','i','j'],'formats':['f','i','i']})
    M1=np.zeros(iterate_left,dtype=Max_Matches) ###初始化保存从左到右最佳匹配的矩阵
    #compute the max match from left to right
    for i in range(iterate_left):
        MAX_COS_DIS=np.max(cos_dis[i,:])
        M1[i]['MAX']=MAX_COS_DIS
        M1[i]['i']=i
        M1[i]['j']=np.argmax(cos_dis[i,:])
    temp_list=(M1,cos_dis)
    return temp_list

def compute_max_match_cc(des1,des2):
    start=time.time()
    (M1,cos_dis)=compute_max_match_nn(des1,des2)
    iterate_left=cos_dis.shape[0]
    iterate_right=cos_dis.shape[1]
    M2=np.zeros(iterate_right,dtype=M1.dtype)
    #compute the max match from right to left
    for j in range(iterate_right):
        MAX_COS_DIS=np.max(cos_dis[:,j])
        M2[j]['MAX']=MAX_COS_DIS
        M2[j]['i']=j
        M2[j]['j']=np.argmax(cos_dis[:,j])
    MAX_XY=[]
    for i in range(iterate_left):
       if M2[M1[i]['j']]['j']==i:
         MAX_XY.append([M1[i]['i'],M1[i]['j'],M1[i]['MAX']])
    MAX_XY=np.array(MAX_XY)
    end=time.time()
    print("compute max match use cc spend total time "+str(end-start)+" seconds")
    return MAX_XY



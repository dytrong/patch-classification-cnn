import numpy as np
import torch
from sklearn import preprocessing
from sklearn.decomposition import PCA
device=torch.device('cuda:1')

####input_data is numpy.array
def pca_whiten(input_data):
    pca = PCA(n_components=input_data.shape[1],whiten=True)
    pca = pca.fit(input_data)
    out_pca= pca.transform(input_data)
    return out_pca

####均值零化操作
def zeroMean(dataMat):
    meanVal = np.mean(dataMat, axis=0)
    newData = dataMat - meanVal
    return newData, meanVal

def ZCAWhitening(dataMat):
    pca=PCA(n_components=dataMat.shape[1],whiten=True)
    PCAWhitening_Data = pca.fit_transform(dataMat)
    newData, meanVal = zeroMean(dataMat) # 数据零均值化,为了求下面的协方差矩阵
    covMat = np.dot(newData.T, newData) / newData.shape[0] # 求协方差矩阵
    U, S, V = np.linalg.svd(covMat) #奇异分解
    ZCAWhitening_Data = PCAWhitening_Data.dot(U.T)
    ZCAWhitening_Data = ZCAWhitening_Data.T
    return ZCAWhitening_Data
#################################
###des1:des1 is a numpy.array####
###des2:des2 is a numpy.array####
def compute_L2_dis(desc1,desc2):
    #desc1=preprocessing.normalize(desc1,norm='l2')
    #print('desc1的形状:'+str(desc1.shape))
    #rank_1=np.linalg.matrix_rank(desc1)
    #print('desc1的秩为:'+str(rank_1))
    #desc2=preprocessing.normalize(desc2,norm='l2')
    #desc1=ZCAWhitening(desc1.T)
    #desc2=ZCAWhitening(desc2.T)
    #print('desc1的形状:'+str(desc1.shape))
    pdist=torch.nn.PairwiseDistance(2)
    desc1=torch.from_numpy(desc1) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    desc2=torch.from_numpy(desc2)
    batch_desc1=torch.chunk(desc1,10,0)       ####torch.chunk(tensor, chunks, dim=0),chunk就是分块的个数,dim=0表示水平方向分隔
    batch_desc2=torch.chunk(desc2,10,0)
    dis_matrix=pdist(batch_desc1[0].cuda(device),batch_desc2[0].cuda(device))  ####第一个子块
    for i in range(1,10):
        temp_dis_matrix=pdist(batch_desc1[i].cuda(device),batch_desc2[i].cuda(device))
        dis_matrix=torch.cat((dis_matrix,temp_dis_matrix),0)   ###torch.cat(inputs, dimension=0)
    return dis_matrix

def compute_cos_dis(desc1,desc2):
    desc1=preprocessing.normalize(desc1,norm='l2')
    desc2=preprocessing.normalize(desc2,norm='l2')
    desc1=torch.from_numpy(desc1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    desc2=torch.from_numpy(desc2).cuda(device)
    dis_matrix=np.zeros((len(desc1),1))
    for i in range(len(desc1)):
        temp_1=torch.dot(desc1[i],desc2[i]) 
        temp_2=torch.pow(torch.dot(desc1[i],desc1[i]),0.5)
        temp_3=torch.pow(torch.dot(desc2[i],desc2[i]),0.5)
        dis_matrix[i]=temp_1/(temp_2*temp_3)
    return dis_matrix

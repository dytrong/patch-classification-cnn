import numpy as np
import torch
device=torch.device('cuda:0')
#################################
###des1:des1 is a numpy.array####
###des2:des2 is a numpy.array####

def compute_L2_dis(desc1,desc2):
    pdist=torch.nn.PairwiseDistance(2)
    desc1=torch.from_numpy(desc1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    desc2=torch.from_numpy(desc2).cuda(device)
    dis_matrix=pdist(desc1,desc2).cpu().numpy()
    return dis_matrix
def compute_cos_dis(desc1,desc2):
    desc1=torch.from_numpy(desc1).cuda(device) ####change numpy.array to torch.tensor and change cpu data to gpu data by use .cuda()
    desc2=torch.from_numpy(desc2).cuda(device)
    dis_matrix=np.zeros((len(desc1),1))
    for i in range(len(desc1)):
        temp_1=torch.dot(desc1[i],desc2[i]).cpu().numpy() 
        temp_2=torch.pow(torch.dot(desc1[i],desc1[i]),0.5).cpu().numpy()
        temp_3=torch.pow(torch.dot(desc2[i],desc2[i]),0.5).cpu().numpy()
        dis_matrix[i]=temp_1/(temp_2*temp_3)
    return dis_matrix

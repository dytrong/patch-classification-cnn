import cv2
import numpy as np
import matplotlib.pyplot as plt

#####分别统计正负样本的欧式距离值
def compute_pos_neg_sample(match_txt_path,dis_matrix):
    pos_distance=[]
    neg_distance=[]
    ground_match_array=np.loadtxt(match_txt_path)
    for i in range(len(ground_match_array)):
        if ground_match_array[i][1]==ground_match_array[i][4]:
            pos_distance.append(dis_matrix[i])
        else:
            neg_distance.append(dis_matrix[i])
    #plt.hist(neg_distance,bins=30,density=1, facecolor='red', alpha=0.75)
    #plt.hist(pos_distance,bins=30,density=1, facecolor='green', alpha=0.75)
    n,bins,patches=plt.hist(pos_distance,bins=30,density=False,stacked=False,label='Positive Sample', facecolor='green', alpha=0.75)
    n1,bins1,patches1=plt.hist(neg_distance,bins=30,density=False,stacked=False,label='Negative Sample',facecolor='red', alpha=0.75)
    print(n)
    print(bins)
    print(patches)
    plt.legend(fontsize=12)
    plt.show()
    #return pos_distance,neg_distance


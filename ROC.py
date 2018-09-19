import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_roc(match_txt_path,dis_matrix,Thresh,match_number):
    ground_match_array=np.loadtxt(match_txt_path)
    ground_correct_number=0
    correct_number=0
    for i in range(len(dis_matrix)):
        sub_path1='./AY/'+str(i)+'.jpg'
        sub_path2='./BY/'+str(i)+'.jpg'
        img1=cv2.imread(sub_path1)
        img2=cv2.imread(sub_path2)
        if dis_matrix[i]<=Thresh:
            correct_number=correct_number+1
            if ground_match_array[i][1]==ground_match_array[i][4]:
                ground_correct_number=ground_correct_number+1
                cv2.imshow('img1',img1)
                cv2.imshow('img2',img2)
                #cv2.waitKey(0)
            else:
                cv2.imshow('img1',img1)
                cv2.imshow('img2',img2)
                #cv2.waitKey(0)
    if correct_number==0:
        return 0,0
    P=float(ground_correct_number/correct_number)
    R=float(ground_correct_number/match_number)
    return P,R

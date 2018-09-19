import numpy as np 
import cv2
match_txt_path='./yosemite_dataset/m50_1000_1000_0.txt'
match_array=np.loadtxt(match_txt_path)
count=0
for i in range(1000):
    if match_array[i][1]==match_array[i][4]:
        count=count+1
        patch_img_path1='./yosemite_dataset/sub_img/'+str(int(match_array[i][0]))+'.jpg'
        patch_img_path2='./yosemite_dataset/sub_img/'+str(int(match_array[i][3]))+'.jpg'
        print(patch_img_path1)
        sub_img1=cv2.imread(patch_img_path1)
        sub_img2=cv2.imread(patch_img_path2)
        #print(sub_img1)
        #print(sub_img1.shape)
        cv2.imshow('sub image 1',sub_img1)
        cv2.imshow('sub image 2',sub_img2)
        cv2.waitKey(0)

print(count)
        

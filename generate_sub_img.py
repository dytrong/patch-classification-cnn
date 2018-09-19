import numpy as np 
import cv2
match_txt_path='./liberty_dataset/m50_10000_10000_0.txt'
match_array=np.loadtxt(match_txt_path)
count=0
for i in range(len(match_array)):
    patch_img_path1='./liberty_dataset/sub_img/'+str(int(match_array[i][0]))+'.jpg'
    patch_img_path2='./liberty_dataset/sub_img/'+str(int(match_array[i][3]))+'.jpg'
    print(patch_img_path1)
    sub_img1=cv2.imread(patch_img_path1)
    sub_img2=cv2.imread(patch_img_path2)
    sub_path1='./AY/'+str(count)+'.jpg'
    sub_path2='./BY/'+str(count)+'.jpg'
    cv2.imwrite(sub_path1,sub_img1)
    cv2.imwrite(sub_path2,sub_img2)
    count=count+1
print(count)
        

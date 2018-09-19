import cv2
def cut_bmp_img(img_path):
    global count
    img=cv2.imread(img_path)
    for i in range(16):
        for j in range(16):
            sub_img=img[i*step:(i+1)*step,j*step:(j+1)*step]
            if sub_img.max()==0:
                return 
            sub_img_path='./yosemite_dataset/sub_img/'+str(count)+'.jpg'
            cv2.imwrite(sub_img_path,sub_img)
            count=count+1
            #cv2.imshow('sub_img',sub_img)
            #cv2.waitKey(0)


if __name__=="__main__":
    step=64
    count=0
    for i in range(2475):
        if i >=0 and i <10:
            bmp_img_path='/home/data1/daizhuang/patch_dataset/yosemite_dataset/patches000'+str(i)+'.bmp'
        if i >=10 and i<100:
            bmp_img_path='/home/data1/daizhuang/patch_dataset/yosemite_dataset/patches00'+str(i)+'.bmp'
        if i>=100 and i<1000:
            bmp_img_path='/home/data1/daizhuang/patch_dataset/yosemite_dataset/patches0'+str(i)+'.bmp' 
        if i>=1000 and i<10000:
            bmp_img_path='/home/data1/daizhuang/patch_dataset/yosemite_dataset/patches'+str(i)+'.bmp'   
        cut_bmp_img(bmp_img_path)


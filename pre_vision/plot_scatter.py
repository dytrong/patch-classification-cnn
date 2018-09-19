import matplotlib.pyplot as plt
import numpy as np 
#txt = ['alexnet','vgg16','vgg19','squezze0','squezze1','res18','res34','res50','res101','res152','inception','dense121','dense169']
#y= [2,0.84,0.5,-0.85,0.22,-1.88,-1.78,-0.88,-1.18,-0.82,5.85,0.32,1.89]
#colors = np.random.rand(13) # 随机产生50个0~1之间的颜色值
#x=[0.99,0.99,0.99,0.21,0.26,0.45,0.57,0.55,0.53,0.55,0.65,0.999,0.999]
plt.xlim(0,14.1)

x1=[12.87]
y1=[2]
x2=[13.86,13.86]
y2=[0.84,0.5]
x3=[2.73,3.37]
y3=[-0.85,0.22]
x4=[3.15,4.55,4.62,3.71,3.85]
y4=[-1.88,-1.78,-0.88,-1.18,-0.82]
x5=[7.8]
y5=[5.85]
x6=[7,7]
y6=[0.32,1.89]

'''
####liberty dataset
x1=[12.87]
y1=[1.86]
x2=[13.86,14]
y2=[1.63,1.63]
x3=[2.47,2.86]
y3=[-0.16,-0.56]
x4=[3.85,4.27,4.41,4.34,4.2]
y4=[-1.46,-1.83,-1.41,-1.74,-1.43]
x5=[7.97]
y5=[5.01]
x6=[7,7]
y6=[1.81,7.63]
'''
'''
####Nd dataset
x1=[12.87]
y1=[0.85]
x2=[13.72,14]
y2=[0.2,0.2]
x3=[2.86,3.25]
y3=[-0.05,-0.46]
x4=[4.06,4.55,4.62,4.76,4.62]
y4=[-0.6,-1.16,-0.88,-0.51,-0.97]
x5=[8.16]
y5=[1.75]
x6=[7,7]
y6=[0.99,5.13]
'''
plt.scatter(x1,y1,c='r',label='alexnet')
plt.scatter(x2,y2,c='b',label='VGG')
plt.scatter(x3,y3,c='g',label='squeze')
plt.scatter(x4,y4,c='y',label='ResNet')
plt.scatter(x5,y5,c='m',label='Inception')
plt.scatter(x6,y6,c='k',label='DenseNet')
plt.plot([0,14.1],[0,0],'r-')
plt.xlabel('The rank of feature map', fontsize=12)
plt.ylabel('Average-Pooling minus Max-Pooling(%)', fontsize=12)
plt.legend(fontsize=12)
plt.show()

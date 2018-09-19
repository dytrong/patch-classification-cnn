import matplotlib.pyplot as plt
x=['2215 vs 0620','2215 vs 1005','2215 vs 1410','2215 vs 1640']
#y1=[0.303,0.299,0.224,0.205]
#y2=[0.442,0.446,0.404,0.352]
#y3=[0.480,0.483,0.439,0.382]
#y4=[0.634,0.633,0.577,0.525]
#y5=[0.584,0.590,0.527,0.474]
#y6=[0.717,0.726,0.666,0.628]
#y7=[0.643,0.645,0.589,0.551]
y1=[63,62,43,38]
y2=[63,40,32,23]
y3=[53,50,41,32]
y4=[74,71,60,51]
y5=[78,75,65,55]
y6=[73,70,58,46]
y7=[76,73,62,53]
plt.plot(x,y7,label='DenseNet121-block2',linewidth=2,color='r',marker='o',markerfacecolor='yellow',markersize=12)
plt.plot(x,y2,label='AlexNet-pool5',linewidth=2,color='blueviolet',marker='h',markerfacecolor='orchid',markersize=12)
plt.plot(x,y6,label='ResNet101-layer2',linewidth=2,color='c',marker='d',markerfacecolor='crimson',markersize=12)
plt.plot(x,y4,label='AlexNet-conv2',linewidth=2,color='lime',marker='^',markerfacecolor='red',markersize=12)
plt.plot(x,y5,label='VGG16-pool3',linewidth=2,color='y',marker='p',markerfacecolor='lime',markersize=12)
plt.plot(x,y3,label='AlexNet-conv5',linewidth=2,color='c',marker='*',markerfacecolor='yellow',markersize=12)
plt.plot(x,y1,label='SIFT',linewidth=2,color='r',marker='H',markerfacecolor='lime',markersize=12)
plt.ylim(10,117)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 设置坐标标签字体大小
plt.xlabel('Different Illumination Changes',fontsize=12)
plt.ylabel('inliner-number',fontsize=12)
plt.legend(fontsize=11)
plt.show() 

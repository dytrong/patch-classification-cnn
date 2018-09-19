import matplotlib.pyplot as plt
x=['2*2','3*3','4*4','5*5','6*6','7*7']
y1=[0.8841,0.8885,0.8890,0.8854,0.8787,0.8705]
y2=[0.9265,0.9259,0.9224,0.9162,0.9081,0.9000]
y3=[0.9365,0.9379,0.9346,0.9291,0.9281,0.9133]
y4=[0.8794,0.8930,0.8945,0.8904,0.8836,0.8750]
y5=[0.9004,0.9003,0.9090,0.9260,0.8994,0.8923]
y6=[0.8832,0.8787,0.8720,0.8634,0.8551,0.8494]
y7=[0.8557,0.8515,0.8470,0.8415,0.8367,0.8342]
y8=[0.8865,0.8859,0.8821,0.8760,0.8702,0.8676]
y9=[0.8904,0.8874,0.8804,0.8725,0.8665,0.8622]
y10=[0.8636,0.8649,0.8622,0.8573,0.8518,0.8462]
y11=[0.9375,0.9580,0.9659,0.9678,0.9674,0.9657]
y12=[0.9402,0.9389,0.9334,0.9275,0.9218,0.9148]
y13=[0.9465,0.9486,0.9447,0.9383,0.9311,0.9191]
y14=[0.9508,0.9498,0.9445,0.9388,0.9330,0.9265]
#plt.plot(x,y1,label='AlexNet',linewidth=2,color='r',marker='o',markerfacecolor='yellow',markersize=12)
#plt.plot(x,y2,label='VGG16',linewidth=2,color='blueviolet',marker='h',markerfacecolor='orchid',markersize=12)
#plt.plot(x,y3,label='VGG19',linewidth=2,color='c',marker='d',markerfacecolor='crimson',markersize=12)
#plt.plot(x,y4,label='SqueezeNet',linewidth=2,color='lime',marker='^',markerfacecolor='red',markersize=12)
#plt.plot(x,y5,label='SqueezeNet1',linewidth=2,color='y',marker='p',markerfacecolor='lime',markersize=12)
#plt.plot(x,y6,label='ResNet18',linewidth=2,color='c',marker='*',markerfacecolor='yellow',markersize=12)
#plt.plot(x,y7,label='ResNet34',linewidth=2,color='r',marker='H',markerfacecolor='lime',markersize=12)

#plt.plot(x,y8,label='ResNet50',linewidth=2,color='r',marker='o',markerfacecolor='yellow',markersize=12)
#plt.plot(x,y9,label='ResNet101',linewidth=2,color='blueviolet',marker='h',markerfacecolor='orchid',markersize=12)
#plt.plot(x,y10,label='ResNet152',linewidth=2,color='c',marker='d',markerfacecolor='crimson',markersize=12)
plt.plot(x,y11,label='Inception-V3',linewidth=2,color='lime',marker='^',markerfacecolor='red',markersize=12)
plt.plot(x,y12,label='DenseNet121',linewidth=2,color='y',marker='p',markerfacecolor='lime',markersize=12)
plt.plot(x,y13,label='DenseNet169',linewidth=2,color='c',marker='*',markerfacecolor='yellow',markersize=12)
#plt.ylim(0.8,0.98) ##resnet
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 设置坐标标签字体大小
plt.xlabel('Pooling Size',fontsize=12)
plt.ylabel('mean Average Precision (mAP)',fontsize=12)
plt.legend(fontsize=11)
plt.show() 

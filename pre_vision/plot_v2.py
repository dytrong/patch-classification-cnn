import matplotlib.pyplot as plt
x=[-40,-30,-20,-10,0,10,20,30,40]
'''
###inlier rate
y1=[0.6878,0.7472,0.7575,0.7419,0.7447,0.7400,0.7677,0.7800,0.7367]
y2=[0.7022,0.7352,0.7538,0.7555,0.7455,0.7664,0.7765,0.7514,0.7534]
y3=[0.7680,0.7605,0.7746,0.7662,0.7439,0.7763,0.7572,0.7400,0.7904]
y4=[0.7135,0.7287,0.7498,0.7729,0.7282,0.7521,0.7857,0.7453,0.7648]
'''
####inlier number
y1=[104,120,123,123,128,125,128,127,115]
y2=[109,120,128,128,130,131,131,127,126]
y3=[112,121,126,125,125,130,123,121,126]
y4=[107,121,126,130,125,128,133,124,126]
'''
'''
###mAP 
y1=[0.8251,0.8585,0.8354,0.8446,0.8555,0.8630,0.8661,0.8579,0.8400]
y2=[0.8902,0.9000,0.8929,0.8987,0.9106,0.9059,0.9058,0.8927,0.9002]
y3=[0.8846,0.9030,0.8914,0.9031,0.9104,0.9030,0.9087,0.8975,0.9054]
y4=[0.8824,0.8980,0.8922,0.8973,0.8973,0.9000,0.9072,0.8870,0.8910]

plt.plot(x,y1,label='AlexNet-conv2',linewidth=2,color='r',marker='o',markerfacecolor='yellow',markersize=12)
plt.plot(x,y2,label='VGG16-pool3',linewidth=2,color='blueviolet',marker='h',markerfacecolor='orchid',markersize=12)
plt.plot(x,y3,label='ResNet101-layer2',linewidth=2,color='c',marker='d',markerfacecolor='crimson',markersize=12)
plt.plot(x,y4,label='DenseNet121-block2',linewidth=2,color='r',marker='H',markerfacecolor='lime',markersize=12)
plt.ylim(0.8,0.965)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
# 设置坐标标签字体大小
plt.xlabel('Intensity change',fontsize=12)
plt.ylabel('mean Average Precision(mAP)',fontsize=12)
plt.legend(fontsize=11)
plt.show() 

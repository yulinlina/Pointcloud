import cv2
import os
os .environ['CUDA_VISIBLE_DEVICES']='0'
import paddlehub as hub
import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

path= 'data/bridge.jpg'
# 模型加载
# use_gpu：是否使用GPU进行预测
model = hub.Module(name='MiDaS_Large', use_gpu=True)

img=cv2.imread(path)
assert len(img)>0
# 模型预测
result = model.depth_estimation(images=[img])
result=result[0]

np.savetxt('data/depthData.txt',result)
out=256*(result-np.min(result))/(np.max(result)-np.min(result))
out=np.array(out,dtype='uint8')
cv2.imwrite('data/depthPic.png',out)
##


sourcePath=path
xmin, ymin, xmax, ymax =  [2740, 772, 2889, 1442]
fig, subs = plt.subplots(1,2)

subs[0].imshow(plt.imread('data/depthPic.png'))
subs[0].add_patch(
     patches.Rectangle(
        (xmin,ymin),
        xmax-xmin,
        ymax-ymin,
        edgecolor = 'red',
        facecolor = 'red',
        fill=False
     ) )
source=plt.imread(sourcePath)

subs[1].imshow(source)
subs[1].add_patch(
     patches.Rectangle(
        (xmin,ymin),
        xmax-xmin,
        ymax-ymin,
        edgecolor = 'red',
        facecolor = 'red',
        fill=False
     ) )

plt.show()
plt.imshow(img)
plt.show()
#2740,772,2889,1442 右側面
#1948,474,2279,1110 左側面
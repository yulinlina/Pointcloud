import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from functionTools import *
from fit import *
from show import *
# 深度图中或点云中选择ROI 区域


# inputType='depthPic'
# inputType='randomGenerate'
def main(inputType='ply',method = "pca"):
    xmin, ymin, xmax, ymax = [2928, 156, 3340, 492]
    xmin, ymin, xmax, ymax = [2740, 772, 2889, 1442]
    # 2740,772,2889,1442 右側面
    # 1948,474,2279,1110 左側面
    xmin, ymin, xmax, ymax = [1948, 474, 2279, 1110]
    if inputType=='depth':
        xmin, ymin, xmax, ymax = [2740, 772, 2889, 1442]
        # 2740,772,2889,1442 右側面
        # 1948,474,2279,1110 左側面
        # xmin, ymin, xmax, ymax = [1948, 474, 2279, 1110]
        x1, y1, z1=getTxtPoint('data/depthData.txt',xmin,ymin,xmax,ymax)
        x2,y2,z2=randomChoice(x1,y1,z1,400)

    elif inputType=='ply':
        path='data/test.ply'
        print('ply files:',path)
        # x1,y1,z1=getPlyPoint(path)
        # plyPlot(x1, y1, z1)
        ##ply points
        xmin, ymin, xmax, ymax = [-0.5, -0.3, -0.178, 0.1]
        # xmin, ymin, xmax, ymax = [-0.178, -0.3, 0.75, 0.1]

        x1,y1,z1=getPlyPoint(path,xmin,ymin,xmax,ymax)
        x2,y2,z2=randomChoice(x1,y1,z1,400)
        #plyPlot(x2, y2, z2)
    print('point nums:',len(z2))

    x2,y2,z2=normalize(x2,y2,z2)

    pointsNum=len(z2)
    xmin,ymin,xmax,ymax=min(x2),min(y2),max(x2),max(y2)

    if method == "pca":
        adata, bdata, cdata, ddata, nVector = pca3D(x2, y2, z2, svdSolver='full')
    elif method=="nn":
        adata, bdata, cdata, ddata = paddleNN(x2, y2, z2, 1)
    elif method =="svd":
        adata,bdata,cdata,ddata=svd(x2,y2,z2)
    elif  method=="sparsepca":
        adata, bdata, cdata, ddata,nVector= pca3D(x2, y2, z2,Sparse=True,svdSolver='full')
    print(f"a: {adata:.3f}, b: {bdata:.3f}, c: {cdata:.3f}, d:{ddata:.3f}")
    print(f'平面拟合结果为：z = {adata:.3f} * x + {bdata:.3f} * y +  {ddata:.3f}')
  #   matplot(adata, bdata, ddata, x2, y2, z2, testNum=10)



if __name__ == "__main__":
    main(inputType='depth',method="nn")



########################
# 引入numpy模块并创建两个三维向量x和y

# print(np.quiver2direction([adata,bdata,1]))
#(21.20529360458012, -26.340669000214305, 34.92197536082232)
#(5.6091964913982935, -35.99981527025057, 36.57354005219686)
##l2 (-38.51978576618727, -3.019308641500217, 38.68279617874105)
##l1 (33.103264906331404, 0.4657960744497558, 33.10740320510461)
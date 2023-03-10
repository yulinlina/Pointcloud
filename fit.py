import numpy as np
import time
from sklearn.decomposition import PCA,KernelPCA,SparsePCA
import matplotlib.pyplot as plt

def paddleNN(x2,y2,z2,testNum=100):
    import paddle
    class MyModel(paddle.nn.Layer):
        def __init__(self):
            super(MyModel, self).__init__()

            a = self.create_parameter([1])
            self.add_parameter("a", a)
            b = self.create_parameter([1])
            self.add_parameter("b", b)
            d = self.create_parameter([1])
            self.add_parameter("d", d)

        def forward(self, inputs):
            x_train = inputs[0]
            y_train = inputs[1]
            z = self.a.expand_as(x_train) * x_train + self.b.expand_as(x_train) * y_train + self.d.expand_as(x_train)

            return z

    x_train=np.array(x2[:-testNum])
    y_train=np.array(y2[:-testNum])
    z_train=np.array(z2[:-testNum])
    x_test=np.array(x2[:-testNum])
    y_test=np.array(y2[:-testNum])
    z_test=np.array(z2[:-testNum])
    model = MyModel()
    model.a.set_value(paddle.to_tensor([-0.66]))
    model.d.set_value(paddle.to_tensor([0.63]))
    model.b.set_value(paddle.to_tensor([0.09]))


    loss_fn = paddle.nn.MSELoss(reduction='mean')


    optimizer = paddle.optimizer.AdamW(weight_decay=0.00002, learning_rate=0.001,
            parameters=model.parameters())


    total_data=len(x_train)
    batch_size=len(x_train)
    loss_list=[]
    for t in range(30000 * (total_data // batch_size)):
        idx = np.random.choice(total_data, batch_size, replace=False)
        xx = paddle.to_tensor(x_train[idx], dtype='float32')
        yy = paddle.to_tensor(y_train[idx], dtype='float32')
        zz = paddle.to_tensor(z_train[idx], dtype='float32')
        pred = model((xx,yy))

        loss = loss_fn(pred, zz)
        loss_list.append(loss)

        if t % 200 == 0:
            print(f"epoch: {t}, loss:{loss.numpy()[0]:.3f}")
            if len(loss_list) > 200:
                if np.average(loss_list[-200:-100]) - np.average(loss_list[-100:]) < 0.0001:
                    print('EARLY STOP')
                    break

        loss.backward()
        optimizer.step()
        optimizer.clear_grad()

    print(model.parameters())
    plt.plot(range(t+1),loss_list)
    plt.legend("loss")
    adata=float(model.parameters()[0].numpy()[0])
    bdata = float(model.parameters()[1].numpy()[0])
    ddata=float(model.parameters()[2].numpy()[0])
    return adata,bdata,-1,ddata

def svd(x2,y2,z2,testNum=100):
    # ??????????????????A
    A = np.zeros((3, 3))
    for i in range(0, len(x2)):
        A[0, 0] = A[0, 0] + x2[i] ** 2
        A[0, 1] = A[0, 1] + x2[i] * y2[i]
        A[0, 2] = A[0, 2] + x2[i]
        A[1, 0] = A[0, 1]
        A[1, 1] = A[1, 1] + y2[i] ** 2
        A[1, 2] = A[1, 2] + y2[i]
        A[2, 0] = A[0, 2]
        A[2, 1] = A[1, 2]
        A[2, 2] = 100
    # print(A)

    # ??????b
    b = np.zeros((3, 1))
    for i in range(0, len(x2)):
        b[0, 0] = b[0, 0] + x2[i] * z2[i]
        b[1, 0] = b[1, 0] + y2[i] * z2[i]
        b[2, 0] = b[2, 0] + z2[i]
    # print(b)

    # ??????X
    A_inv = np.linalg.inv(A)
    X = np.dot(A_inv, b)
    # print('????????????????????????z = %.3f * x + %.3f * y + %.3f' % (X[0, 0], X[1, 0], X[2, 0]))

    # ????????????
    R = 0
    for i in range(0, len(x2)):
        R = R + (X[0, 0] * x2[i] + X[1, 0] * y2[i] + X[2, 0] - z2[i]) ** 2
    print('????????????%.*f' % (3, R))
    return X[0, 0], X[1, 0],-1, X[2, 0]


from scipy.spatial.distance import pdist,squareform
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pca3D(xx, yy, zz, svdSolver='full', Sparse=False,points=None, showFlag=True):
    if points is None:
        points = np.c_[xx, yy, zz]

    t1 = time.time()
    ## ??????3????????????
    if Sparse:
        pcaPlane =SparsePCA(n_components=3)
    else:
        pcaPlane = PCA(n_components=3, svd_solver=svdSolver)
    pcaPlane.fit(points)

    V= pcaPlane.components_.T
    x1, y1, z1 = pcaPlane.components_[0]
    x2, y2, z2 = pcaPlane.components_[1]

    print(V)

    x_pca_axis, y_pca_axis, z_pca_axis = 3 * V
    print('pca time', time.time() - t1)
    x_pca_plane = np.r_[x_pca_axis[:2], -x_pca_axis[1::-1]]
    y_pca_plane = np.r_[y_pca_axis[:2], -y_pca_axis[1::-1]]
    z_pca_plane = np.r_[z_pca_axis[:2], -z_pca_axis[1::-1]]
    x_pca_plane.shape = (2, 2)
    y_pca_plane.shape = (2, 2)
    z_pca_plane.shape = (2, 2)

    # ??????1 ??? ???2 ????????????????????????????????????????????????????????????3??????

    x3, y3, z3 = np.average(xx), np.average(yy), np.average(zz)

    ###
    if showFlag:
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        ox, oy, oz = x3, y3, z3
        fig = plt.figure(1)
        plt.clf()
        ax = Axes3D(fig)

        ax.scatter(xx[::10], yy[::10], zz[::10], marker="+", alpha=0.8)
        surfaceFlag = True
        if surfaceFlag:
            ax.plot_surface(x_pca_plane, y_pca_plane, z_pca_plane,
                            color='r',
                            alpha=0.5)
        else:
            ax.plot_wireframe(x_pca_plane, y_pca_plane, z_pca_plane,
                              rstride=10, cstride=10, color='r')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        # for i, (comp, var) in enumerate(zip(pcaPlane.components_, pcaPlane.explained_variance_)):
        #     comp = comp * (1 - var)  # scale component by its variance explanation power
        #     ax.quiver(ox, oy, oz, comp[0] + ox, comp[1] + oy, comp[2] + oz,
        #               length=0.5, normalize=True,
        #               color=f"C{i + 2}", )

        plt.show()

    aData = ((y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1))
    bData = ((z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1))
    cData = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
    dData = 0 - (aData * x1 + bData * y1 + cData * z1)
    nVector = -1 * np.cross(np.array([(x1 - x3), y1 - y3, z1 - z3]),
                            np.array([(x2 - x3), y2 - y3, z2 - z3]))

    return aData / cData, bData / cData, cData / cData, dData / cData, nVector
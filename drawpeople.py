import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
post_out1=np.zeros((10,3))
post_out1[:,:2]=np.load('./test2/moxing20_2d_134.npy')[0].reshape(10,2)
f1=open('D:\大四下\继续毕设\kinect\\test2\\test2\\data_2024505095600.txt')
lines=f1.readlines()[0:25]
post_out3=np.zeros((25,3))
index=0
for item in lines:
    posX=float(item.split(' ')[0])
    posY=float(item.split(' ')[1])
    posZ=float(item.split(' ')[2][:-1])
    post_out3[index][0]=posX
    post_out3[index][1] = posZ
    post_out3[index][2] = posY
    index+=1


post_out2=np.zeros((11,3))
# post_out2[0]=post_out3[3]
# post_out2[1]=post_out3[21]
# post_out2[2]=post_out3[4]
# post_out2[3]=post_out3[20]
# post_out2[4]=post_out3[8]
# post_out2[5]=post_out3[23]
# post_out2[6]=post_out3[15]
# post_out2[7]=post_out3[12]
# post_out2[8]=post_out3[0]
# post_out2[9]=post_out3[16]
# post_out2[10]=post_out3[19]
# for i in range(0,11):
#     post_out2[i][1]=0

post_out2[0]=post_out1[0]
post_out2[3]=post_out2[1]
post_out2[2]=post_out1[2]+post_out2[3]
post_out2[1]=post_out1[1]+post_out2[2]
post_out2[4]=-post_out1[3]+post_out2[3]
post_out2[5]=-post_out1[4]+post_out2[4]
post_out2[8]=-post_out1[5]+post_out2[3]
post_out2[7]=post_out1[7]+post_out2[8]
post_out2[6]=post_out1[6]+post_out2[7]
post_out2[9]=-post_out1[8]+post_out2[8]
post_out2[10]=-post_out1[9]+post_out2[9]
# post_out = np.array([
#     [0, 0, 0.787822068],
#     [0.123238757, -0.040780518, 0.78164047],
#     [0.131162703, 0.05052188, 0.356339693],
#     [0.101998687, -0.134696901, 0],
#     [-0.123237714, 0.040785622, 0.79400897],
#     [-0.040740184, 0.075435936, 0.358784974],
#     [-0.032220125, -0.093336225, 0.009030402],
#     [-0.004283257, -0.027451605, 1.036530614],
#     [-0.020511895, -0.014475852, 1.293020725],
#     [-0.035262831, 0.071317613, 1.387965202],
#     [-0.07418853, 0.023139238, 1.495238662],
#     [-0.176504895, 0.024071693, 1.260654569],
#     [-0.167135298, 0.210691005, 1.104242444],
#     [0.009177908, 0.251894534, 1.199225664],
#     [0.130504638, -0.098547608, 1.273252487],
#     [0.233572304, -0.229689911, 1.055113673],
#     [0.307354271, -0.038980789, 0.936925113],
# ])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def show3Dpose(vals, ax):
    ax.view_init(elev=34., azim=70)

    lcolor = (0, 0, 1)
    rcolor = (1, 0, 0)

    # I = np.array([0, 0, 1, 4, 2, 5, 0, 7, 8, 8, 14, 15, 11, 12, 8, 9])
    # J = np.array([1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    I = np.array([0, 3, 1, 3, 4, 3, 8, 7, 8, 9])
    J = np.array([3, 2, 2, 4, 5, 8, 7, 6, 9, 10])# 头，左臂，右臂，躯干，左腿，右腿
    # I = np.array([2, 20, 20, 4, 6, 6, 21, 22, 8, 9, 9, 10,   10, 1, 14, 0, 0, 0, 12, 13,16,17,18])
    # J = np.array([3, 2, 4,   5, 5, 7, 7, 6,   20, 8, 10, 11, 24, 20, 15, 12, 16, 1,13, 14, 17,18,19])

    # LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1,0, 1, 0, 1, 0, 1, 0, 0, 0, 1,0,0,1], dtype=bool)
    LR = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)

    for i in np.arange(len(I)):
        # x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, color=lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])
    ax.set_zlim3d([-RADIUS_Z + zroot, RADIUS_Z + zroot])
    ax.set_aspect('auto')  # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)
    ax.grid(False) # 去除网格线
    ax.tick_params('x', labelbottom=False)
    ax.tick_params('y', labelleft=False)
    ax.tick_params('z', labelleft=False)


show3Dpose(post_out2, ax)
plt.show()
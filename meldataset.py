from torch.utils import data
import numpy as np
from PIL import Image
import scipy.io as scio
import torch
import torch.nn.functional as F

class Mat_dataset(data.Dataset):
    def __init__(self,file_path):
        self.file_path = file_path
        f=open("train2.txt","r")
        f_list=f.readlines()
        self.files=f_list
        # f_dict={}
        # for item in f_list:
        #     f_dict[item.split(' ')[0]]=int(item.split(' ')[1][:-1])
        # self.label_dict=f_dict
        # L_name = 'D:\大四下\继续毕设\kinect\\test2\\test2\\data_2024505' + self.files[index - 1][:-1].split('0506')[
        #     -1] + '.txt'
        # file_t = open(L_name)
        # lines = file_t.readlines()
        # self.ll=int(len(lines) / 26)
        f.close()

    def __getitem2__(self,index):
        A_name = 'D:\大四上\FineVib\\network\\be314\\'+self.files[index-1][:-1]+'_A.mat'
        D_name = 'D:\大四上\FineVib\\network\\be314\\' + self.files[index - 1][:-1] + '_D.mat'
        L_name='D:\大四下\继续毕设\kinect\\test2\\test2\\data_2024505'+ self.files[index - 1][:-1].split('0506')[-1]+'.txt'
        mat_dataA = scio.loadmat(A_name)['RAs']
        mat_dataD = scio.loadmat(D_name)['RDs']
        t1 = torch.tensor(mat_dataA)
        t2 = torch.tensor(mat_dataD)
        file_t=open(L_name)
        lines=file_t.readlines()
        legses=[]
        for i in range(0, int(len(lines) / 26)):
            f_temp = lines[26 * i:26 * i + 25]
            bones = []
            for it in f_temp:
                x1 = float(it.split(' ')[0])
                y1 = float(it.split(' ')[1])
                z1 = float(it.split(' ')[2])
                xt = [x1, y1, z1]
                bones.append(xt)
            n_bones2 = np.array(bones)
            n_bones = np.zeros((25,2))
            n_bones[:, 0] = n_bones2[:, 0]
            n_bones[:,1]=n_bones2[:,2]

            leg1 = n_bones[3] - n_bones[20]
            leg2 = n_bones[21] - n_bones[4]
            leg3 = n_bones[4] - n_bones[20]
            leg4 = n_bones[20] - n_bones[8]
            leg5 = n_bones[8] - n_bones[23]
            leg6 = n_bones[20] - n_bones[0]
            leg7 = n_bones[15] - n_bones[12]
            leg8 = n_bones[12] - n_bones[0]
            leg9 = n_bones[0] - n_bones[16]
            leg10 = n_bones[16] - n_bones[19]
            # 注意，为了降低拟合难度，抛弃了一维坐标
            legs = np.concatenate((leg1,leg2, leg3, leg4, leg5, leg6, leg7, leg8, leg9, leg10), axis=0)

            legses.append(legs)
        le=torch.tensor(np.array(legses))
        t1=t1[:len(legses)]
        t2 = t2[:len(legses)]
        t3=torch.cat((t1,t2),2)
        # 存储npy文件
        fs = open('train2.txt', 'a')
        for i in range(0,t3.shape[0]):
            np.save('dataset/RAs/'+self.files[index-1][:-1]+'_A'+str(i)+'.npy',t1[i].float().numpy())
            np.save('dataset/RDs/' + self.files[index - 1][:-1] + '_D' + str(i) + '.npy', t2[i].float().numpy())
            np.save('dataset/ground/' + self.files[index - 1][:-1] + '_g' + str(i) + '.npy', le[i].float().numpy())
            fs.write(self.files[index - 1][:-1] + '_' + str(i) + '\n')
        return t3.float().cuda(),le.float().cuda() # 降低数据精度

    def __getitemD__(self,index):# DNet版本

        A_name = 'D:\大四下\继续毕设\mmwave感知训练\dataset\RAs\\'+self.files[index-1][:19]+'A'+self.files[index-1][19:-1]+'.npy'
        D_name = 'D:\大四下\继续毕设\mmwave感知训练\dataset\RDs\\'+self.files[index-1][:19]+'D'+self.files[index-1][19:-1]+'.npy'
        L_name='D:\大四下\继续毕设\mmwave感知训练\dataset\ground\\'+self.files[index-1][:19]+'g'+self.files[index-1][19:-1]+'.npy'
        mat_dataA = np.load(A_name)
        mat_dataD = np.load(D_name)
        ground_truth=np.load(L_name)
        t1 = torch.tensor(mat_dataA)
        t2 = torch.tensor(mat_dataD)
        le1=torch.tensor(ground_truth)
        t3=torch.cat((t1,t2),1)
        t4=F.normalize(t3.reshape(16,16,308),dim=0)
        le=F.normalize(le1,dim=0)
        return t4.cuda(),le.cuda() # 降低数据精度,[256,308] [20]

    def __getitem__(self,index):# ENet版本

        A_name = './dataset/RAs/'+self.files[index-1][:19]+'A'+self.files[index-1][19:-1]+'.npy' # 服务器可用
        D_name = './dataset/RDs/'+self.files[index-1][:19]+'D'+self.files[index-1][19:-1]+'.npy'
        L_name='./dataset/ground/'+self.files[index-1][:19]+'g'+self.files[index-1][19:-1]+'.npy'
        mat_dataA = np.load(A_name)
        mat_dataD = np.load(D_name)
        ground_truth=np.load(L_name)
        t1 = torch.tensor(mat_dataA)
        t2 = torch.tensor(mat_dataD)# [256,180]和[256,128]
        le1=torch.tensor(ground_truth)
        t3 = F.normalize(t1.reshape(4, 128, 90), dim=0)
        t4 = F.normalize(t2.reshape(4, 128, 64), dim=0)
        le = F.normalize(le1, dim=0)
        return (t3.cuda(),t4.cuda()),le.cuda() # 降低数据精度,[256,308] [20]

    def __len__(self):
        return len(self.files)


# f=open("train2.txt","r")
# f_list=f.readlines()
# f_dict={}
# for item in f_list:
#     f_dict[item.split(' ')[0]]=int(item.split(' ')[1][:-1])
# print(f_dict)